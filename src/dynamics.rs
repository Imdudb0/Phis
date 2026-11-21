use crate::graph::*;

pub const L_MAX: usize = 48;

#[derive(Clone)]
pub struct Assembly { pub span: (usize, usize), pub ws: Vec<S>, pub nodes: Vec<usize>, pub sig: Vec<f32> }
#[derive(Clone)]
pub struct Link { pub a: usize, pub b: usize, pub score: f32, pub shared: bool }
pub struct Elig(pub Vec<[f32; D * D]>);

pub struct SearchResult {
    pub best_c: Vec<S>, pub best_ws: Vec<S>, pub best_e: Vec<u8>, pub best_probs: Vec<[f32; 256]>,
    pub phi: f32, pub fidelity: f32, pub phi_trace: Vec<f32>, pub cycles: usize, pub ops: u64,
    pub assemblies: Vec<Assembly>, pub links: Vec<Link>, pub residual_mean: f32,
}

pub fn chunk_e(e: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if e.is_empty() { out.push((0, 0)); return out; }
    let win = ((e.len() / 16).max(24)).min(64);
    let mut s = 0usize;
    for i in 0..=e.len() {
        let end = i == e.len();
        let sep = !end && matches!(e[i], b' ' | b'\n' | b'\t' | b'\r' | b'|' | b'?' | b'.' | b';' | b',');
        if end || sep {
            if i > s { out.push((s, i)); }
            s = i + 1;
        } else if i - s >= win {
            out.push((s, i)); s = i;
        }
    }
    if out.is_empty() { out.push((0, e.len())); }
    while out.len() > 24 {
        let mut bi = 0; let mut bs = usize::MAX;
        for i in 0..out.len() - 1 {
            let sz = (out[i].1 - out[i].0) + (out[i + 1].1 - out[i + 1].0);
            if sz < bs { bs = sz; bi = i; }
        }
        let merged = (out[bi].0, out[bi + 1].1);
        out[bi] = merged; out.remove(bi + 1);
    }
    out
}

// Perception: E stimule l'alphabet; le régional R intègre.
pub fn perceive(g: &Graph, e: &[u8], rng: &mut Rng, budget: &mut Budget) -> (Vec<S>, Vec<Assembly>) {
    let mut x = zero_state();
    let gain = vec![1.0f32; N];
    let zero = zero_state();
    let mut assemblies = Vec::new();
    for &(s, t) in &chunk_e(e) {
        for p in s..t {
            let b = e[p] as usize;
            let mut input = zero_state();
            input[b][0] = 1.0;
            for _ in 0..2 { x = step_full(g, &x, &input, &gain, 0.0, rng, budget, None); }
        }
        for _ in 0..4 { x = step_full(g, &x, &zero, &gain, 0.0, rng, budget, None); }
        let ws: Vec<S> = x[WS0..C0].to_vec();
        let mut idx: Vec<usize> = (0..N_WS).collect();
        idx.sort_by(|&a, &b| norm(&ws[b]).partial_cmp(&norm(&ws[a])).unwrap());
        idx.truncate(4);
        assemblies.push(Assembly { span: (s, t), ws, nodes: idx, sig: Vec::new() });
    }
    (x, assemblies)
}

// Régénération: la cause candidate ré-exprime E par l'alphabet.
pub fn regenerate(g: &Graph, ws: &[S], c0: &[S], l: usize, noise: f32,
                  cut: Option<EdgeKind>, lesion: Option<usize>,
                  rng: &mut Rng, budget: &mut Budget) -> (Vec<u8>, Vec<[f32; 256]>) {
    let mut x = zero_state();
    x[WS0..C0].copy_from_slice(ws);
    x[C0..G0].copy_from_slice(c0);
    let gain = vec![1.0f32; N];
    let mut ep = Vec::with_capacity(l);
    let mut probs = Vec::with_capacity(l);
    for _ in 0..l {
        if let Some(k) = lesion { x[C0 + k] = [0.0; D]; }
        x = gen_step(g, &x, &gain, noise, rng, budget, cut);
        if let Some(k) = lesion { x[C0 + k] = [0.0; D]; }
        let p = softmax_bytes(&x[0..N_BYTE]);
        ep.push(argmax_byte(&p));
        probs.push(p);
    }
    (ep, probs)
}

pub fn fidelity(ep: &[u8], e: &[u8], l: usize) -> f32 {
    if l == 0 { return 0.0; }
    let mut c = 0;
    for p in 0..l { if ep[p] == e[p] { c += 1; } }
    c as f32 / l as f32
}
pub fn residual_mean(probs: &[[f32; 256]], e: &[u8], l: usize) -> f32 {
    if l == 0 { return 1.0; }
    let mut s = 0.0;
    for p in 0..l { s += 1.0 - probs[p][e[p] as usize]; }
    s / l as f32
}

fn worst_assembly(asms: &[Assembly], probs: &[[f32; 256]], e: &[u8], l: usize) -> Option<usize> {
    let mut best = None; let mut bv = -1.0;
    for (i, a) in asms.iter().enumerate() {
        let (s, t) = a.span; let mut acc = 0.0; let mut n = 0;
        for p in s..t.min(l) { acc += 1.0 - probs[p][e[p] as usize]; n += 1; }
        let v = if n > 0 { acc / n as f32 } else { 0.0 };
        if v > bv { bv = v; best = Some(i); }
    }
    best
}

// La recherche: hypothèse -> régénération -> résidu -> modulation/création.
// Le scaffold est le méta-a-priori; la trajectoire dépend du sens trouvé.
pub fn search(g: &mut Graph, e: &[u8], max_cycles: usize, budget: &mut Budget,
              rng: &mut Rng, eta_fast: f32, explore_gain: f32,
              mut elig: Option<&mut Elig>) -> SearchResult {
    let l = e.len().min(L_MAX);
    let (mut x, mut assemblies) = perceive(g, e, rng, budget);
    let zero = zero_state(); let gain1 = vec![1.0f32; N];
    for _ in 0..8 { x = step_full(g, &x, &zero, &gain1, 0.0, rng, budget, None); }
    let sigs = crate::abstraction::signatures(g, &assemblies, rng, budget);
    for (a, s) in assemblies.iter_mut().zip(sigs.into_iter()) { a.sig = s; }

    let mut best_phi = -1.0f32; let mut best_fid = 0.0f32; let mut best_resid = 1.0f32;
    let mut best_c = x[C0..G0].to_vec(); let mut best_ws = x[WS0..C0].to_vec();
    let mut best_e = vec![0u8; l]; let mut best_probs = vec![[0.0; 256]; l];
    let mut phi_prev = 0.0f32; let mut resid_prev = 1.0f32; let mut stall = 0usize;
    let mut phi_trace = Vec::new(); let mut cycles = 0usize;
    let mut last_phi = 0.0f32; let mut last_fid = 0.0f32; let mut last_resid = 1.0f32;

    for cycle in 0..max_cycles {
        if budget.ops >= budget.max { break; }
        cycles = cycle + 1;
        let ws = x[WS0..C0].to_vec(); let c = x[C0..G0].to_vec();
        let (ep, probs) = regenerate(g, &ws, &c, l, 0.0, None, None, rng, budget);
        let fid = fidelity(&ep, e, l);
        let resid = residual_mean(&probs, e, l);
        last_fid = fid; last_resid = resid;

        let mut phi = last_phi;
        let phi_now = cycle % 3 == 0;
        let reward = phi - phi_prev;
        if phi_now {
            phi = crate::phi::compute(g, &ws, &c, e, rng, budget);
            last_phi = phi; phi_trace.push(phi);
            if phi > best_phi {
                best_phi = phi; best_fid = fid; best_resid = resid;
                best_c = c.clone(); best_ws = ws.clone(); best_e = ep.clone(); best_probs = probs.clone();
            }
            if phi > phi_prev + 1e-4 { stall = 0; } else { stall += 1; }
            phi_prev = phi;
        } else if best_phi < 0.0 && fid > best_fid {
            best_fid = fid; best_c = c.clone(); best_ws = ws.clone(); best_e = ep.clone(); best_probs = probs.clone();
        }

        // Éligibilité trois-facteurs (lente, entraînement seulement).
        let dr = (resid_prev - resid).max(0.0);
        if let Some(el) = elig.as_deref_mut() {
            for (ei, ed) in g.edges.iter().enumerate() {
                if ed.kind.is_fast() { continue; }
                let o = outer(&x[ed.src as usize], &x[ed.dst as usize]);
                for k in 0..D * D { el.0[ei][k] += o[k] * dr; }
            }
        }
        // Plasticité rapide intrinsèque (marche aussi à l'inférence, zéro-shot).
        if eta_fast > 0.0 && reward.abs() > 1e-9 {
            for ed in g.edges.iter_mut() {
                if !ed.kind.is_fast() { continue; }
                let o = outer(&x[ed.src as usize], &x[ed.dst as usize]);
                for k in 0..D * D { ed.w[k] = (ed.w[k] + eta_fast * reward * o[k]).clamp(-2.0, 2.0); }
            }
        }
        // Abduction créative: sur stagnation, le résidu engendre une NOUVELLE liaison.
        if stall >= 3 {
            if let Some(wi) = worst_assembly(&assemblies, &probs, e, l) {
                let ck = (0..N_C).max_by(|&a, &b|
                    norm(&x[C0 + a]).partial_cmp(&norm(&x[C0 + b])).unwrap()).unwrap_or(0);
                let targets: Vec<usize> = assemblies[wi].nodes.iter().take(2).map(|&i| WS0 + i).collect();
                for dst in targets {
                    let src = C0 + ck;
                    let eis: Vec<usize> = g.incoming[dst].iter().copied().filter(|&ei| {
                        let ed = &g.edges[ei]; ed.kind == EdgeKind::CWs && ed.src as usize == src
                    }).collect();
                    if !eis.is_empty() {
                        for ei in eis {
                            let o = outer(&x[src], &x[dst]);
                            for k in 0..D * D {
                                g.edges[ei].w[k] = (g.edges[ei].w[k] + eta_fast * 0.5 * o[k]).clamp(-2.0, 2.0);
                            }
                        }
                    } else {
                        g.add_edge(src, dst, EdgeKind::CWs, rng, 0.05);
                    }
                }
            }
            stall = 0;
        }

        // La procédure de recherche dépend du sens trouvé: température pilotée par le résidu.
        let temp = ((0.05 + 0.25 * resid + 0.03 * stall as f32) * explore_gain).min(0.6);
        let mut gain = vec![1.0f32; N];
        for k in 0..N_C { gain[C0 + k] = 1.0 + 0.5 * resid; }
        for b in 0..N_BYTE { gain[b] = 0.6; }
        for _ in 0..6 { x = step_full(g, &x, &zero, &gain, temp, rng, budget, None); }
        resid_prev = resid;
        if resid < 0.02 && stall > 2 { break; }
    }

    if best_phi < 0.0 { best_phi = last_phi; best_fid = last_fid; best_resid = last_resid; }
    let links = crate::abstraction::project(g, &mut assemblies, e, rng, budget);
    SearchResult {
        best_c, best_ws, best_e, best_probs,
        phi: best_phi.max(0.0), fidelity: best_fid, phi_trace, cycles, ops: budget.ops,
        assemblies, links, residual_mean: best_resid,
    }
}