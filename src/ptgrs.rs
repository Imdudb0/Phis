use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CauseFamily {
    Periodic,
    Symmetric,
    Nested,
    SharedMotif,
    Substitution,
    Grammar,
}

impl CauseFamily {
    pub const ALL: [CauseFamily; 6] = [
        CauseFamily::Periodic,
        CauseFamily::Symmetric,
        CauseFamily::Nested,
        CauseFamily::SharedMotif,
        CauseFamily::Substitution,
        CauseFamily::Grammar,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CauseFamily::Periodic => "periodic",
            CauseFamily::Symmetric => "symmetric",
            CauseFamily::Nested => "nested",
            CauseFamily::SharedMotif => "shared",
            CauseFamily::Substitution => "substitution",
            CauseFamily::Grammar => "grammar",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "periodic" => Some(CauseFamily::Periodic),
            "symmetric" => Some(CauseFamily::Symmetric),
            "nested" => Some(CauseFamily::Nested),
            "shared" => Some(CauseFamily::SharedMotif),
            "substitution" => Some(CauseFamily::Substitution),
            "grammar" => Some(CauseFamily::Grammar),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SepMode {
    None,
    Newline,
    Pipe,
    Random,
}

#[derive(Clone, Debug)]
pub struct Episode {
    pub effect: Vec<u8>,
    pub hidden_cause: CauseFamily,
}

/// Épisode composite en flux continu. Les métadonnées `hidden_*` ne servent
/// qu'au diagnostic post-hoc et ne sont jamais vues pendant l'entraînement.
#[derive(Clone, Debug)]
pub struct CompositeEpisode {
    pub effect: Vec<u8>,
    pub hidden_query_idx: usize,
    pub hidden_match_idx: usize,
    pub hidden_families: Vec<CauseFamily>,
    pub hidden_core_spans: Vec<(usize, usize)>,
    pub hidden_tag_spans: Vec<Option<(usize, usize)>>,
}

struct TermNode {
    terminal: Option<u8>,
    children: Vec<usize>,
}

struct TermGraph {
    nodes: Vec<TermNode>,
}

impl TermGraph {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn add_leaf(&mut self, byte: u8) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TermNode {
            terminal: Some(byte),
            children: Vec::new(),
        });
        id
    }

    fn add_branch(&mut self, children: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TermNode {
            terminal: None,
            children,
        });
        id
    }

    fn linearize(&self, root: usize, max_len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(max_len);
        self.dfs(root, 0, max_len, &mut out);
        out
    }

    fn dfs(&self, id: usize, depth: usize, max_len: usize, out: &mut Vec<u8>) {
        if out.len() >= max_len || depth > 28 {
            return;
        }
        let node = &self.nodes[id];
        if let Some(b) = node.terminal {
            out.push(b);
            return;
        }
        for &child in &node.children {
            self.dfs(child, depth + 1, max_len, out);
            if out.len() >= max_len {
                break;
            }
        }
    }
}

fn sample_alphabet(rng: &mut Rng, count: usize, lo: u8, hi: u8) -> Vec<u8> {
    let mut pool: Vec<u8> = (lo..=hi).collect();
    rng.shuffle(&mut pool);
    pool.truncate(count.min(pool.len()));
    pool
}

pub fn sample_episode(rng: &mut Rng, family: CauseFamily) -> Episode {
    sample_episode_in_range(rng, family, 32, 126)
}

pub fn sample_episode_in_range(rng: &mut Rng, family: CauseFamily, lo: u8, hi: u8) -> Episode {
    let mut tg = TermGraph::new();
    let max_len = 28 + rng.next_usize(21);

    let root = match family {
        CauseFamily::Periodic => {
            let period = 2 + rng.next_usize(4);
            let alpha = sample_alphabet(rng, period, lo, hi);
            let leaves: Vec<usize> = alpha.into_iter().map(|b| tg.add_leaf(b)).collect();
            let motif = tg.add_branch(leaves);
            let reps = (max_len / period).max(6) + 1;
            tg.add_branch(vec![motif; reps])
        }
        CauseFamily::Symmetric => {
            let half = 3 + rng.next_usize(4);
            let odd = rng.next_usize(2) == 1;
            let alpha = sample_alphabet(rng, half + usize::from(odd), lo, hi);
            let fwd: Vec<usize> = alpha[..half].iter().map(|&b| tg.add_leaf(b)).collect();
            let rev: Vec<usize> = fwd.iter().rev().copied().collect();
            let mut kids = fwd;
            if odd {
                kids.push(tg.add_leaf(alpha[half]));
            }
            kids.extend(rev);
            let pal = tg.add_branch(kids);
            tg.add_branch(vec![pal, pal, pal, pal])
        }
        CauseFamily::Nested => {
            let depth = 4 + rng.next_usize(3);
            let alpha = sample_alphabet(rng, depth * 2 + 2, lo, hi);
            let core_a = tg.add_leaf(alpha[0]);
            let core_b = tg.add_leaf(alpha[1]);
            let mut cur = tg.add_branch(vec![core_a, core_b, core_a]);
            for d in 0..depth {
                let open = tg.add_leaf(alpha[2 + d * 2]);
                let close = tg.add_leaf(alpha[3 + d * 2]);
                cur = tg.add_branch(vec![open, cur, close]);
            }
            let r = cur;
            tg.add_branch(vec![r, r, r])
        }
        CauseFamily::SharedMotif => {
            let alpha = sample_alphabet(rng, 8, lo, hi);
            let m0 = tg.add_leaf(alpha[0]);
            let m1 = tg.add_leaf(alpha[1]);
            let m2 = tg.add_leaf(alpha[2]);
            let shared = tg.add_branch(vec![m0, m1, m2, m1]);
            let b0 = tg.add_leaf(alpha[3]);
            let b1 = tg.add_leaf(alpha[4]);
            let b2 = tg.add_leaf(alpha[5]);
            let b3 = tg.add_leaf(alpha[6]);
            let b4 = tg.add_leaf(alpha[7]);
            tg.add_branch(vec![
                shared, b0, shared, b1, shared, b2, shared, b3, shared, b4, shared,
            ])
        }
        CauseFamily::Substitution => {
            let alpha = sample_alphabet(rng, 5, lo, hi);
            let a0 = tg.add_leaf(alpha[0]);
            let a1 = tg.add_leaf(alpha[1]);
            let a2 = tg.add_leaf(alpha[2]);
            let a3 = tg.add_leaf(alpha[3]);
            let a4 = tg.add_leaf(alpha[4]);
            let mut prev = tg.add_branch(vec![a0, a1]);
            let mut curr = tg.add_branch(vec![a2, a3, a4]);
            for _ in 0..6 {
                let next = tg.add_branch(vec![curr, prev]);
                prev = curr;
                curr = next;
            }
            curr
        }
        CauseFamily::Grammar => {
            let alpha = sample_alphabet(rng, 14, lo, hi);
            let sep = tg.add_leaf(alpha[0]);

            let r0 = tg.add_leaf(alpha[1]);
            let r1 = tg.add_leaf(alpha[2]);
            let r2 = tg.add_leaf(alpha[3]);
            let stem = tg.add_branch(vec![r0, r1, r2]);

            let s0 = tg.add_leaf(alpha[4]);
            let s1 = tg.add_leaf(alpha[5]);
            let w0 = tg.add_branch(vec![stem, s0]);
            let w1 = tg.add_branch(vec![stem, s1]);

            let a0 = tg.add_leaf(alpha[6]);
            let a1 = tg.add_leaf(alpha[7]);
            let a2 = tg.add_leaf(alpha[8]);
            let w2 = tg.add_branch(vec![a0, a1, a2]);

            let b0 = tg.add_leaf(alpha[9]);
            let b1 = tg.add_leaf(alpha[10]);
            let b2 = tg.add_leaf(alpha[11]);
            let w3 = tg.add_branch(vec![b0, b1, b2]);

            let clause_a = tg.add_branch(vec![w0, sep, w2, sep, w1]);
            let clause_b = tg.add_branch(vec![w2, sep, w3, sep, w0]);
            tg.add_branch(vec![clause_a, sep, clause_b, sep, clause_a])
        }
    };

    let mut effect = tg.linearize(root, max_len);
    while effect.len() < 24 {
        let copy = effect.clone();
        effect.extend_from_slice(&copy);
    }
    effect.truncate(max_len);

    Episode {
        effect,
        hidden_cause: family,
    }
}

fn choose_sep(rng: &mut Rng, mode: SepMode) -> &'static [u8] {
    match mode {
        SepMode::None => b"",
        SepMode::Newline => b"\n",
        SepMode::Pipe => b" | ",
        SepMode::Random => match rng.next_usize(4) {
            0 => b"",
            1 => b"\n",
            2 => b" | ",
            _ => b"; ",
        },
    }
}

pub fn sample_composite_episode(
    rng: &mut Rng,
    num_molds: usize,
    with_tags: bool,
    printable_query: bool,
    query_last: bool,
    sep_mode: SepMode,
) -> CompositeEpisode {
    let mut fams = CauseFamily::ALL.to_vec();
    rng.shuffle(&mut fams);
    let k = num_molds.clamp(2, fams.len());
    fams.truncate(k);

    let target_fam = fams[0];
    let sep = choose_sep(rng, sep_mode);

    // Chaque entrée : (tag_optionnel, corps_causal, famille, is_query, is_match)
    let mut entries: Vec<(Option<Vec<u8>>, Vec<u8>, CauseFamily, bool, bool)> =
        Vec::with_capacity(k + 1);

    for (idx, &fam) in fams.iter().enumerate() {
        let body = sample_episode_in_range(rng, fam, 33, 90).effect;
        let tag = if with_tags {
            let prefix = if printable_query {
                format!("MOLD_{}: ", fam.name().to_uppercase()).into_bytes()
            } else {
                let tag_len = 8 + rng.next_usize(5);
                sample_alphabet(rng, tag_len, 33, 90)
            };
            Some(prefix)
        } else {
            None
        };
        entries.push((tag, body, fam, false, idx == 0));
    }

    let (q_lo, q_hi) = if printable_query {
        (97, 122)
    } else {
        (140, 220)
    };
    let query = sample_episode_in_range(rng, target_fam, q_lo, q_hi).effect;

    if query_last {
        rng.shuffle(&mut entries);
        entries.push((None, query, target_fam, true, false));
    } else {
        entries.push((None, query, target_fam, true, false));
        rng.shuffle(&mut entries);
    }

    let mut effect = Vec::new();
    let mut hidden_query_idx = 0;
    let mut hidden_match_idx = 0;
    let mut hidden_families = Vec::with_capacity(entries.len());
    let mut hidden_core_spans = Vec::with_capacity(entries.len());
    let mut hidden_tag_spans = Vec::with_capacity(entries.len());

    for (i, (tag_opt, body, fam, is_query, is_match)) in entries.into_iter().enumerate() {
        if i > 0 {
            effect.extend_from_slice(sep);
        }
        let tag_span = if let Some(tag) = tag_opt {
            let ts = effect.len();
            effect.extend_from_slice(&tag);
            let te = effect.len();
            Some((ts, te))
        } else {
            None
        };
        let cs = effect.len();
        effect.extend_from_slice(&body);
        let ce = effect.len();

        if is_query {
            hidden_query_idx = i;
        }
        if is_match {
            hidden_match_idx = i;
        }
        hidden_families.push(fam);
        hidden_core_spans.push((cs, ce));
        hidden_tag_spans.push(tag_span);
    }

    CompositeEpisode {
        effect,
        hidden_query_idx,
        hidden_match_idx,
        hidden_families,
        hidden_core_spans,
        hidden_tag_spans,
    }
}
