use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

type Res<T> = Result<T, Box<dyn Error>>;

const N: usize = 64;
const REGIONS: usize = 4;
const SETTLING: usize = 12;
const MAGIC: &[u8] = b"PHISHARP";
const VERSION: u32 = 1;
const LN_256: f32 = 256.0_f32.ln();

/* ------------------------------------------------------------------------- */
/* Aléatoire interne                                                         */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1_u32 << 24) as f32)
    }

    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    fn usize(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn bool(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

/* ------------------------------------------------------------------------- */
/* Utilitaires vectoriels                                                    */
/* ------------------------------------------------------------------------- */

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt().max(1.0e-8)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b) / (norm(a) * norm(b))
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

fn softmax(logits: &mut [f32]) {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;

    for x in logits.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }

    for x in logits {
        *x /= sum.max(1.0e-12);
    }
}

fn full_mask() -> u64 {
    if N == 64 {
        u64::MAX
    } else {
        (1_u64 << N) - 1
    }
}

fn across(mask: u64, i: usize, j: usize) -> bool {
    (((mask >> i) ^ (mask >> j)) & 1) != 0
}

/* ------------------------------------------------------------------------- */
/* Dictionnaire                                                              */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
struct Entry {
    term: String,
    definition: String,
}

fn load_dictionary(path: &str) -> Res<Vec<Entry>> {
    let text = fs::read_to_string(path)?;
    let mut entries = Vec::new();

    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let pair = line
            .split_once('\t')
            .or_else(|| line.split_once(" ::= "))
            .or_else(|| line.split_once(" = "));

        let Some((term, definition)) = pair else {
            return Err(format!(
                "{}:{} : entrée attendue sous la forme terme<TAB>définition",
                path,
                line_no + 1
            )
            .into());
        };

        if term.trim().is_empty() || definition.trim().is_empty() {
            return Err(format!("{}:{} : entrée vide", path, line_no + 1).into());
        }

        entries.push(Entry {
            term: term.trim().to_string(),
            definition: definition.trim().to_string(),
        });
    }

    if entries.is_empty() {
        return Err("dictionnaire vide".into());
    }

    Ok(entries)
}

/* ------------------------------------------------------------------------- */
/* Graphe connexionniste non réciproque                                      */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
struct Graph {
    /// W : transition causale dirigée.
    w: Vec<f32>,

    /// U : seconde transition dirigée, apprise indépendamment de W.
    /// W[i,j] n'est jamais lié à W[j,i], U[i,j] ou U[j,i].
    u: Vec<f32>,

    bias: Vec<f32>,

    /// Projection octet -> état causal.
    encoder: Vec<f32>,

    /// Régénération état -> octet. Elle ne sert jamais de modèle de langue.
    decoder: Vec<f32>,
}

impl Graph {
    fn new(rng: &mut Rng) -> Self {
        let mut w = vec![0.0; N * N];
        let mut u = vec![0.0; N * N];
        let mut encoder = vec![0.0; 256 * N];
        let mut decoder = vec![0.0; N * 256];

        for i in 0..N {
            for j in 0..N {
                w[i * N + j] = rng.signed() * 0.075;
                u[i * N + j] = rng.signed() * 0.045;
            }
        }

        for x in &mut encoder {
            *x = rng.signed() * 0.18;
        }

        for x in &mut decoder {
            *x = rng.signed() * 0.012;
        }

        Self {
            w,
            u,
            bias: vec![0.0; N],
            encoder,
            decoder,
        }
    }

    fn step(&self, x: &[f32], z: &[f32], cut: Option<u64>) -> Vec<f32> {
        let mut y = vec![0.0; N];

        for i in 0..N {
            let mut a = self.bias[i] + 0.22 * z[i];

            for j in 0..N {
                if cut.is_some_and(|m| across(m, i, j)) {
                    continue;
                }

                let k = i * N + j;
                a += self.w[k] * x[j];
                a += self.u[k] * x[j] * (0.5 + 0.5 * x[i]);
            }

            y[i] = a.tanh();
        }

        y
    }

    fn settle(&self, z: &[f32], cut: Option<u64>) -> Vec<f32> {
        let mut x: Vec<f32> = z.iter().map(|v| (0.35 * v).tanh()).collect();

        for _ in 0..SETTLING {
            let y = self.step(&x, z, cut);

            for i in 0..N {
                x[i] = 0.42 * x[i] + 0.58 * y[i];
            }
        }

        x
    }

    /// L'alphabet est toujours celui des octets présents dans E.
    fn encode(&self, effect: &[u8]) -> Vec<f32> {
        let mut z = vec![0.0; N];
        let mut total = 0.0;

        for (p, &byte) in effect.iter().enumerate() {
            let temporal = 1.0 / (1.0 + p as f32 / 48.0);
            total += temporal;

            for i in 0..N {
                let phase = ((p + 1) * (i + 1)) as f32 * 0.00091;
                let order = 1.0 + 0.17 * phase.sin();
                z[i] += temporal * order * self.encoder[byte as usize * N + i];
            }
        }

        if total > 0.0 {
            for v in &mut z {
                *v /= total;
            }
        }

        z
    }

    fn state_at(x: &[f32], position: usize) -> Vec<f32> {
        let mut s = x.to_vec();

        for i in 0..N {
            let t = (position + 1) as f32;
            let f = (i + 1) as f32;
            s[i] += 0.055 * (t * f * 0.0017).sin();
            s[i] += 0.035 * (t * (N - i) as f32 * 0.0023).cos();
        }

        s
    }

    fn distribution(&self, x: &[f32], position: usize) -> [f32; 256] {
        let s = Self::state_at(x, position);
        let mut logits = [0.0_f32; 256];

        for byte in 0..256 {
            logits[byte] = dot(
                &self.decoder[byte * N..(byte + 1) * N],
                &s,
            ) * 0.7;
        }

        softmax(&mut logits);
        logits
    }

    fn loss(&self, effect: &[u8], x: &[f32]) -> f32 {
        if effect.is_empty() {
            return 0.0;
        }

        effect
            .iter()
            .enumerate()
            .map(|(p, &byte)| {
                let q = self.distribution(x, p);
                -q[byte as usize].max(1.0e-12).ln()
            })
            .sum::<f32>()
            / effect.len() as f32
    }

    fn effect_repertoire(&self, effect: &[u8], x: &[f32]) -> [f32; 256] {
        let mut mean = [0.0_f32; 256];

        if effect.is_empty() {
            mean.fill(1.0 / 256.0);
            return mean;
        }

        let count = effect.len().min(16);
        let stride = (effect.len() / count).max(1);
        let mut used = 0;

        for p in (0..effect.len()).step_by(stride).take(count) {
            let q = self.distribution(x, p);

            for b in 0..256 {
                mean[b] += q[b];
            }

            used += 1;
        }

        for q in &mut mean {
            *q /= used.max(1) as f32;
        }

        mean
    }

    /// Descente sur la seule régénération de E, puis consolidation des arêtes
    /// traversant la partition minimale. Aucune cause cachée n'intervient.
    fn learn(&mut self, effect: &[u8], h: &Candidate, lr: f32) {
        if effect.is_empty() {
            return;
        }

        let positions = effect.len().min(32);
        let stride = (effect.len() / positions).max(1);
        let scale = 1.0 / positions as f32;
        let mut gx = vec![0.0_f32; N];

        for p in (0..effect.len()).step_by(stride).take(positions) {
            let target = effect[p] as usize;
            let s = Self::state_at(&h.x, p);
            let q = self.distribution(&h.x, p);

            for byte in 0..256 {
                let error = (q[byte] - f32::from(byte == target)) * scale;

                for i in 0..N {
                    let k = byte * N + i;
                    gx[i] += self.decoder[k] * error;
                    self.decoder[k] -= lr * error * s[i];
                    self.decoder[k] = self.decoder[k].clamp(-3.0, 3.0);
                }
            }
        }

        let y = self.step(&h.x, &h.z, None);

        for i in 0..N {
            let gp = gx[i] * (1.0 - y[i] * y[i]);
            self.bias[i] -= lr * 0.15 * gp;

            for j in 0..N {
                let k = i * N + j;
                let cross = across(h.cut, i, j);

                self.w[k] -= lr * 0.08 * gp * h.x[j];
                self.u[k] -= lr * 0.04 * gp * h.x[j] * (0.5 + 0.5 * h.x[i]);

                if cross {
                    let integration = lr * 0.0025 * h.phi * h.x[i] * h.x[j];
                    self.w[k] += integration;
                    self.u[k] -= integration * 0.37;
                }

                self.w[k] = self.w[k].clamp(-1.5, 1.5);
                self.u[k] = self.u[k].clamp(-1.5, 1.5);
            }
        }

        for &byte in effect {
            for i in 0..N {
                let k = byte as usize * N + i;
                let pull = lr * 0.0008 * (h.x[i] - self.encoder[k]);
                self.encoder[k] += pull;
                self.encoder[k] = self.encoder[k].clamp(-2.0, 2.0);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* φ_sharp                                                                   */
/* ------------------------------------------------------------------------- */

#[derive(Clone, Copy)]
struct Phi {
    value: f32,
    cut: u64,
    cause: f32,
    effect: f32,
    integration: f32,
}

fn partitions() -> Vec<u64> {
    let mut result = vec![
        0xaaaa_aaaa_aaaa_aaaa,
        0xcccc_cccc_cccc_cccc,
        0xf0f0_f0f0_f0f0_f0f0,
        0xff00_ff00_ff00_ff00,
        0xffff_0000_ffff_0000,
        0x0000_0000_ffff_ffff,
    ];

    for k in 0..10_u64 {
        let mut mask = 0_u64;

        for i in 0..N {
            let bit = ((i as u64 * 17 + k * 29 + (i as u64 / 7) * 11) % 23) < 11;
            if bit {
                mask |= 1_u64 << i;
            }
        }

        result.push(mask & full_mask());
    }

    result.retain(|&m| m != 0 && m != full_mask());
    result
}

impl Graph {
    fn phi_sharp(&self, effect: &[u8], z: &[f32], x: &[f32]) -> Phi {
        let whole = self.step(x, z, None);
        let q_whole = self.effect_repertoire(effect, x);

        let loss = self.loss(effect, x);
        let effect_info = (1.0 - loss / LN_256).clamp(0.0, 1.0);

        let uniform = 1.0 / 256.0;
        let repertoire_info = q_whole
            .iter()
            .map(|q| (q - uniform).abs())
            .sum::<f32>()
            * 0.5;

        let effect_power = (0.72 * effect_info + 0.28 * repertoire_info).clamp(0.0, 1.0);

        let mut cause_power = 0.0;

        for region in 0..REGIONS {
            let mut perturbed = x.clone();

            for i in 0..N {
                if i * REGIONS / N == region {
                    perturbed[i] = -perturbed[i];
                }
            }

            let counterfactual = self.step(&perturbed, z, None);
            cause_power += distance(&whole, &counterfactual);
        }

        cause_power = (cause_power / REGIONS as f32 * 2.0).tanh();

        let mut mip = f32::INFINITY;
        let mut min_cut = 0;

        for mask in partitions() {
            let partitioned = self.step(x, z, Some(mask));
            let cause_distance = distance(&whole, &partitioned);
            let q_partitioned = self.effect_repertoire(effect, &partitioned);

            let effect_distance = q_whole
                .iter()
                .zip(q_partitioned.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                * 0.5;

            let integrated = (cause_distance * 1.8 + effect_distance).clamp(0.0, 1.0);

            if integrated < mip {
                mip = integrated;
                min_cut = mask;
            }
        }

        if !mip.is_finite() {
            mip = 0.0;
        }

        let value = (cause_power * effect_power * mip)
            .max(0.0)
            .cbrt();

        Phi {
            value,
            cut: min_cut,
            cause: cause_power,
            effect: effect_power,
            integration: mip,
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Recherche abductive                                                       */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
struct Candidate {
    z: Vec<f32>,
    x: Vec<f32>,
    loss: f32,
    phi: f32,
    cut: u64,
    score: f32,
}

struct Abduction {
    best: Candidate,
    evaluations: usize,
    phi: Phi,
}

impl Graph {
    fn evaluate(&self, effect: &[u8], z: Vec<f32>) -> Candidate {
        let x = self.settle(&z, None);
        let loss = self.loss(effect, &x);
        let phi = self.phi_sharp(effect, &z, &x);
        let regeneration = (1.0 - loss / LN_256).clamp(-1.0, 1.0);

        Candidate {
            score: 3.4 * phi.value + regeneration - 0.035 * loss,
            z,
            x,
            loss,
            phi: phi.value,
            cut: phi.cut,
        }
    }

    fn mutate(z: &[f32], rng: &mut Rng, sigma: f32) -> Vec<f32> {
        z.iter()
            .map(|v| {
                let a = rng.signed();
                let b = rng.signed();
                (v + sigma * (a + b) * 0.5).clamp(-3.0, 3.0)
            })
            .collect()
    }

    fn abduce(&self, effect: &[u8], entries: &[Entry], budget: usize, rng: &mut Rng) -> Abduction {
        let budget = budget.max(1);
        let base = self.encode(effect);

        let mut projected: Vec<(f32, usize)> = entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let bytes = format!("{} {}", entry.term, entry.definition);
                (cosine(&base, &self.encode(bytes.as_bytes())), i)
            })
            .collect();

        projected.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        let mut population = vec![base.clone()];

        for &(_, i) in projected.iter().take(5) {
            let bytes = format!("{} {}", entries[i].term, entries[i].definition);
            let projection = self.encode(bytes.as_bytes());
            let mut seed = base.clone();

            for k in 0..N {
                seed[k] = 0.52 * seed[k] + 0.48 * projection[k];
            }

            population.push(seed);
        }

        while population.len() < 12 {
            population.push(Self::mutate(&base, rng, 0.32));
        }

        let mut evaluations = 0;
        let mut best: Option<Candidate> = None;
        let mut generation = 0;

        while evaluations < budget {
            let mut scored = Vec::new();

            for z in population {
                if evaluations >= budget {
                    break;
                }

                let candidate = self.evaluate(effect, z);
                evaluations += 1;

                if best
                    .as_ref()
                    .is_none_or(|b| candidate.score > b.score)
                {
                    best = Some(candidate.clone());
                }

                scored.push(candidate);
            }

            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            let elites: Vec<Candidate> = scored.into_iter().take(6).collect();
            let mut next: Vec<Vec<f32>> = elites.iter().map(|c| c.z.clone()).collect();
            let sigma = 0.34 / (1.0 + generation as f32 * 0.09);

            while next.len() < 14 && !elites.is_empty() {
                let parent = &elites[rng.usize(elites.len())].z;
                next.push(Self::mutate(parent, rng, sigma));
            }

            population = next;
            generation += 1;
        }

        let best = best.expect("au moins une hypothèse");
        let phi = self.phi_sharp(effect, &best.z, &best.x);

        Abduction {
            best,
            evaluations,
            phi,
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Abstraction méta-relationnelle et projection référentielle                */
/* ------------------------------------------------------------------------- */

fn relational_signature(x: &[f32]) -> Vec<f32> {
    let mut signature = vec![0.0; REGIONS * 8];

    for i in 0..N {
        let r = i * REGIONS / N;
        signature[r * 8] += x[i];
        signature[r * 8 + 1] += x[i] * x[i];
        signature[r * 8 + 2] += x[i].tanh();

        let j = (i * 29 + 17) % N;
        signature[r * 8 + 3] += x[i] * x[j];
        signature[r * 8 + 4] += (x[i] - x[j]).abs();

        let k = (i + 1) % N;
        signature[r * 8 + 5] += x[i] * x[k];
        signature[r * 8 + 6] += (x[i] + x[j]) * 0.5;
        signature[r * 8 + 7] += (x[i] - x[j]).sin();
    }

    for chunk in signature.chunks_exact_mut(8) {
        for value in chunk {
            *value /= N as f32 / REGIONS as f32;
        }
    }

    signature
}

#[derive(Clone)]
struct Referent {
    index: usize,
    score: f32,
    direct: f32,
    relational: f32,
}

fn project_referents(graph: &Graph, entries: &[Entry], x: &[f32]) -> Vec<Referent> {
    let signature = relational_signature(x);
    let mut result = Vec::new();

    for (index, entry) in entries.iter().enumerate() {
        let bytes = format!("{} {}", entry.term, entry.definition);
        let z = graph.encode(bytes.as_bytes());
        let projected = graph.settle(&z, None);

        let direct = cosine(x, &projected);
        let relational = cosine(&signature, &relational_signature(&projected));

        let score = 0.38 * direct + 0.62 * relational;

        result.push(Referent {
            index,
            score,
            direct,
            relational,
        });
    }

    result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    result
}

/* ------------------------------------------------------------------------- */
/* Verbalisation maximale et résidu                                          */
/* ------------------------------------------------------------------------- */

fn residual(effect: &[u8], graph: &Graph, h: &Abduction, referents: &[Referent]) -> String {
    let mut unclear: Vec<(f32, usize)> = effect
        .iter()
        .enumerate()
        .map(|(p, &byte)| {
            let q = graph.distribution(&h.best.x, p);
            (-q[byte as usize].max(1.0e-12).ln(), p)
        })
        .collect();

    unclear.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut parts = Vec::new();

    if !unclear.is_empty() && unclear[0].0 > LN_256 * 0.55 {
        let locations = unclear
            .iter()
            .take(8)
            .map(|(_, p)| format!("octet {}={:02x}", p, effect[*p]))
            .collect::<Vec<_>>()
            .join(", ");

        parts.push(format!(
            "une contrainte causale supplémentaire est requise autour de {}",
            locations
        ));
    }

    if h.phi.cause < 0.12 {
        parts.push(
            "plusieurs causes régionales restent interchangeables ; une variation observable de leurs conditions de production manque"
                .to_string(),
        );
    }

    if h.phi.integration < 0.025 {
        parts.push(
            "les relations entre les parties de perceived.eff sont sous-déterminées ; il manque une interaction ou un ordre observable"
                .to_string(),
        );
    }

    if referents
        .first()
        .is_none_or(|r| r.score < 0.12)
    {
        parts.push(
            "aucun référent défini ne porte encore assez la cause globale ; une définition couvrant cette relation manque"
                .to_string(),
        );
    }

    if h.best.loss > LN_256 * 0.65 {
        parts.push(
            "la régénération laisse des effets sans cause commune ; des octets discriminants ou un contexte de production sont requis"
                .to_string(),
        );
    }

    if parts.is_empty() {
        "aucun résidu régional saillant ; les contraintes présentes suffisent à la cause retenue".to_string()
    } else {
        parts.join(" ; ")
    }
}

fn explain(
    effect: &[u8],
    model: &Model,
    h: &Abduction,
) -> (String, Vec<Referent>) {
    let referents = project_referents(&model.graph, &model.entries, &h.best.x);
    let best_score = referents.first().map_or(0.0, |r| r.score);

    let selected: Vec<&Referent> = referents
        .iter()
        .filter(|r| r.score >= best_score - 0.085 && r.score > 0.015)
        .collect();

    let selected: Vec<&Referent> = if selected.is_empty() {
        referents.iter().take(1).collect()
    } else {
        selected
    };

    let mut out = String::new();

    out.push_str(&format!(
        "La recherche abductive converge après {} évaluations causales : φ_sharp={:.6}, puissance de cause={:.6}, puissance d'effet={:.6}, intégration={:.6}, résidu de régénération={:.6}.\n\n",
        h.evaluations,
        h.phi.value,
        h.phi.cause,
        h.phi.effect,
        h.phi.integration,
        h.best.loss
    ));

    out.push_str("Sens découvert et élaboration maximale :\n");

    for (rank, r) in selected.iter().enumerate() {
        let entry = &model.entries[r.index];

        out.push_str(&format!(
            "{}. {} — {} [projection référentielle={:.4}, similarité d'état={:.4], abstraction méta-relationnelle={:.4}]\n",
            rank + 1,
            entry.term,
            entry.definition,
            r.score,
            r.direct,
            r.relational
        ));
    }

    if selected.len() > 1 {
        out.push_str("\nRelations supportées simultanément par la cause :\n");

        for pair in selected.windows(2) {
            let a = &model.entries[pair[0].index];
            let b = &model.entries[pair[1].index];

            let relation = if pair[0].relational >= pair[0].direct.abs() {
                "partagent une organisation relationnelle indépendante de leurs mots"
            } else {
                "sont projetés dans deux aspects d'une même cause globale"
            };

            out.push_str(&format!("- {} et {} {}.\n", a.term, b.term, relation));
        }
    }

    out.push_str(&format!(
        "\nRésidu de perceived.eff : {}.\n",
        residual(effect, &model.graph, h, &referents)
    ));

    (out, referents)
}

/* ------------------------------------------------------------------------- */
/* PTGRS                                                                     */
/* ------------------------------------------------------------------------- */

#[derive(Clone)]
enum Term {
    Leaf(usize),
    Cat(Box<Term>, Box<Term>),
    Relation(u8, Box<Term>, Box<Term>),
    Echo(Box<Term>),
}

impl Term {
    fn show(&self, entries: &[Entry]) -> String {
        match self {
            Self::Leaf(i) => format!("Émettre({})", entries[*i].term),
            Self::Cat(a, b) => format!("Composer({}, {})", a.show(entries), b.show(entries)),
            Self::Relation(r, a, b) => format!(
                "Relation{}({}, {})",
                r,
                a.show(entries),
                b.show(entries)
            ),
            Self::Echo(a) => format!("Réitérer({})", a.show(entries)),
        }
    }
}

struct Ptgrs;

impl Ptgrs {
    fn sample(entries: &[Entry], rng: &mut Rng, depth: usize) -> Term {
        if depth == 0 || rng.bool(0.46) {
            return Term::Leaf(rng.usize(entries.len()));
        }

        match rng.usize(10) {
            0..=4 => Term::Cat(
                Box::new(Self::sample(entries, rng, depth - 1)),
                Box::new(Self::sample(entries, rng, depth - 1)),
            ),
            5..=8 => Term::Relation(
                rng.usize(4) as u8,
                Box::new(Self::sample(entries, rng, depth - 1)),
                Box::new(Self::sample(entries, rng, depth - 1)),
            ),
            _ => Term::Echo(Box::new(Self::sample(entries, rng, depth - 1))),
        }
    }

    fn rewrite(term: &Term, entries: &[Entry], rng: &mut Rng, out: &mut String) {
        match term {
            Term::Leaf(i) => {
                let entry = &entries[*i];

                if rng.bool(0.76) {
                    out.push_str(&entry.definition);
                } else {
                    out.push_str(&entry.term);
                }
            }

            Term::Cat(a, b) => {
                Self::rewrite(a, entries, rng, out);
                out.push(' ');
                Self::rewrite(b, entries, rng, out);
            }

            Term::Relation(kind, a, b) => {
                const LINKS: [&str; 4] = [
                    " qui ",
                    " parce que ",
                    " tandis que ",
                    " dont le sens dépend de ",
                ];

                Self::rewrite(a, entries, rng, out);
                out.push_str(LINKS[*kind as usize % LINKS.len()]);
                Self::rewrite(b, entries, rng, out);
            }

            Term::Echo(a) => {
                let mut local = String::new();
                Self::rewrite(a, entries, rng, &mut local);
                out.push_str(&local);

                if !local.is_empty() {
                    out.push_str(" ; ");
                    out.push_str(local.trim());
                }
            }
        }
    }

    /// La cause est retournée uniquement pour gen/eval. train la détruit
    /// immédiatement et ne transmet que E au système.
    fn generate(entries: &[Entry], rng: &mut Rng) -> (Vec<u8>, Term) {
        let cause = Self::sample(entries, rng, 3);
        let mut effect = String::new();
        Self::rewrite(&cause, entries, rng, &mut effect);

        while effect.len() > 4096 {
            let mut cut = 4096;
            while !effect.is_char_boundary(cut) {
                cut -= 1;
            }
            effect.truncate(cut);
        }

        (effect.into_bytes(), cause)
    }
}

/* ------------------------------------------------------------------------- */
/* Modèle et sérialisation                                                   */
/* ------------------------------------------------------------------------- */

struct Model {
    graph: Graph,
    entries: Vec<Entry>,
    inference_budget: usize,
    trained_steps: usize,
    seed: u64,
}

impl Model {
    fn new(entries: Vec<Entry>, seed: u64) -> Self {
        let mut rng = Rng::new(seed);

        Self {
            graph: Graph::new(&mut rng),
            entries,
            inference_budget: 128,
            trained_steps: 0,
            seed,
        }
    }

    fn save(&self, path: &str) -> Res<()> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        put_u32(&mut out, VERSION);
        put_u64(&mut out, self.seed);
        put_u64(&mut out, self.inference_budget as u64);
        put_u64(&mut out, self.trained_steps as u64);
        put_u32(&mut out, N as u32);

        put_f32_slice(&mut out, &self.graph.w);
        put_f32_slice(&mut out, &self.graph.u);
        put_f32_slice(&mut out, &self.graph.bias);
        put_f32_slice(&mut out, &self.graph.encoder);
        put_f32_slice(&mut out, &self.graph.decoder);

        put_u32(&mut out, self.entries.len() as u32);

        for entry in &self.entries {
            put_string(&mut out, &entry.term);
            put_string(&mut out, &entry.definition);
        }

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        fs::write(path, out)?;
        Ok(())
    }

    fn load(path: &str) -> Res<Self> {
        let bytes = fs::read(path)?;
        let mut c = Cursor::new(&bytes);

        if c.take(MAGIC.len())? != MAGIC {
            return Err("ce fichier n'est pas un modèle Phisharp".into());
        }

        if c.u32()? != VERSION {
            return Err("version de modèle incompatible".into());
        }

        let seed = c.u64()?;
        let inference_budget = c.u64()? as usize;
        let trained_steps = c.u64()? as usize;
        let n = c.u32()? as usize;

        if n != N {
            return Err(format!("dimension de graphe incompatible : {}", n).into());
        }

        let graph = Graph {
            w: c.f32_slice(N * N)?,
            u: c.f32_slice(N * N)?,
            bias: c.f32_slice(N)?,
            encoder: c.f32_slice(256 * N)?,
            decoder: c.f32_slice(N * 256)?,
        };

        let count = c.u32()? as usize;
        let mut entries = Vec::with_capacity(count);

        for _ in 0..count {
            entries.push(Entry {
                term: c.string()?,
                definition: c.string()?,
            });
        }

        Ok(Self {
            graph,
            entries,
            inference_budget,
            trained_steps,
            seed,
        })
    }
}

fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_f32_slice(out: &mut Vec<u8>, values: &[f32]) {
    put_u64(out, values.len() as u64);

    for x in values {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self
            .position
            .checked_add(n)
            .ok_or("dépassement de lecture")?;

        if end > self.bytes.len() {
            return Err("modèle tronqué".into());
        }

        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f32_slice(&mut self, expected: usize) -> Res<Vec<f32>> {
        let count = self.u64()? as usize;

        if count != expected {
            return Err("taille de matrice invalide".into());
        }

        let mut values = Vec::with_capacity(count);

        for _ in 0..count {
            values.push(f32::from_le_bytes(self.take(4)?.try_into()?));
        }

        Ok(values)
    }

    fn string(&mut self) -> Res<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
}

/* ------------------------------------------------------------------------- */
/* Commandes                                                                 */
/* ------------------------------------------------------------------------- */

fn option(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn required(args: &[String], name: &str) -> Res<String> {
    option(args, name).ok_or_else(|| format!("argument requis : {}", name).into())
}

fn train(args: &[String]) -> Res<()> {
    let model_path = required(args, "--model")?;
    let dictionary = required(args, "--dictionary")?;
    let steps = option(args, "--steps")
        .unwrap_or_else(|| "192".into())
        .parse::<usize>()?;
    let seed = option(args, "--seed")
        .unwrap_or_else(|| "0x50a1a5".into())
        .replace("0x", "");
    let seed = u64::from_str_radix(&seed, 16).or_else(|_| seed.parse::<u64>())?;

    let entries = load_dictionary(&dictionary)?;
    let mut model = Model::new(entries, seed);
    let mut rng = Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15);

    println!(
        "Séance abductive unique : {} définitions, {} effets synthétiques.",
        model.entries.len(),
        steps
    );

    let mut mean_phi = 0.0;
    let mut mean_loss = 0.0;

    for step in 0..steps {
        let effect = if step % 5 == 0 {
            let i = rng.usize(model.entries.len());
            model.entries[i].definition.as_bytes().to_vec()
        } else {
            let (effect, hidden_cause) = Ptgrs::generate(&model.entries, &mut rng);
            drop(hidden_cause);
            effect
        };

        if effect.is_empty() {
            continue;
        }

        let search_budget = 18;
        let h = model
            .graph
            .abduce(&effect, &model.entries, search_budget, &mut rng);

        mean_phi += h.best.phi;
        mean_loss += h.best.loss;

        let lr = 0.055 / (1.0 + step as f32 / (steps.max(1) as f32 * 0.7));
        model.graph.learn(&effect, &h.best, lr);
        model.trained_steps += 1;

        if (step + 1) % 16 == 0 || step + 1 == steps {
            let n = (step + 1) as f32;
            println!(
                "étape {:>5}/{:<5}  φ_moyen={:.6}  résidu_moyen={:.6}",
                step + 1,
                steps,
                mean_phi / n,
                mean_loss / n
            );
        }
    }

    model.save(&model_path)?;
    println!("Modèle écrit : {}", model_path);
    Ok(())
}

fn generate(args: &[String]) -> Res<()> {
    let dictionary = required(args, "--dictionary")?;
    let file = required(args, "--file")?;
    let entries = load_dictionary(&dictionary)?;
    let seed = option(args, "--seed")
        .unwrap_or_else(|| "17".into())
        .parse::<u64>()?;
    let mut rng = Rng::new(seed);

    let (effect, cause) = Ptgrs::generate(&entries, &mut rng);
    fs::write(&file, effect)?;

    if let Some(path) = option(args, "--cause") {
        fs::write(path, cause.show(&entries))?;
    }

    println!("E écrit dans {}", file);
    Ok(())
}

fn ask(args: &[String]) -> Res<()> {
    let model_path = required(args, "--model")?;
    let file = required(args, "--file")?;
    let model = Model::load(&model_path)?;
    let effect = fs::read(&file)?;

    if effect.is_empty() {
        return Err("perceived.eff est vide : aucun effet à expliquer".into());
    }

    let mut rng = Rng::new(model.seed ^ effect.len() as u64 ^ 0xa8bd_105e_3379);
    let h = model.graph.abduce(
        &effect,
        &model.entries,
        model.inference_budget,
        &mut rng,
    );

    let (text, _) = explain(&effect, &model, &h);
    print!("{}", text);
    Ok(())
}

fn evaluate(args: &[String]) -> Res<()> {
    let model_path = required(args, "--model")?;
    let file = required(args, "--file")?;
    let cause_path = required(args, "--cause")?;

    let model = Model::load(&model_path)?;
    let effect = fs::read(file)?;
    let cause = fs::read_to_string(cause_path)?;
    let mut rng = Rng::new(model.seed ^ 0xe7a1);

    let h = model.graph.abduce(
        &effect,
        &model.entries,
        model.inference_budget,
        &mut rng,
    );

    let (text, referents) = explain(&effect, &model, &h);
    print!("{}", text);

    if let Some(top) = referents.first() {
        let term = &model.entries[top.index].term;
        println!(
            "\nÉvaluation a posteriori seulement : référent={} ; présent_dans_cause_cachée={}",
            term,
            cause.contains(term)
        );
    }

    Ok(())
}

fn usage() {
    eprintln!(
        "Phisharp\n\
         \n\
         train --model out/sharp.phis --dictionary definitions.fr [--steps 192] [--seed 1]\n\
         gen --dictionary definitions.fr --file perceived.eff [--cause hidden.cause] [--seed 17]\n\
         ask --model out/sharp.phis --file perceived.eff\n\
         eval --model out/sharp.phis --file perceived.eff --cause hidden.cause\n\
         \n\
         Dictionnaire : une entrée par ligne, terme<TAB>définition."
    );
}

fn run() -> Res<()> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        Some("train") => train(&args),
        Some("gen") => generate(&args),
        Some("ask") => ask(&args),
        Some("eval") => evaluate(&args),
        _ => {
            usage();
            Ok(())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("phisharp: {}", error);
        std::process::exit(2);
    }
}
