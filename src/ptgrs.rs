use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CauseFamily {
    Periodic,
    Symmetric,
    Nested,
    SharedMotif,
    Arithmetic,
}

impl CauseFamily {
    pub const ALL: [CauseFamily; 5] = [
        CauseFamily::Periodic,
        CauseFamily::Symmetric,
        CauseFamily::Nested,
        CauseFamily::SharedMotif,
        CauseFamily::Arithmetic,
    ];

    pub fn name(self) -> &'static str {
        match self {
            CauseFamily::Periodic => "periodic",
            CauseFamily::Symmetric => "symmetric",
            CauseFamily::Nested => "nested",
            CauseFamily::SharedMotif => "shared",
            CauseFamily::Arithmetic => "arithmetic",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "periodic" => Some(CauseFamily::Periodic),
            "symmetric" => Some(CauseFamily::Symmetric),
            "nested" => Some(CauseFamily::Nested),
            "shared" => Some(CauseFamily::SharedMotif),
            "arithmetic" => Some(CauseFamily::Arithmetic),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Episode {
    pub effect: Vec<u8>,
    pub hidden_cause: CauseFamily,
}

struct TermNode {
    terminal: Option<u8>,
    children: Vec<usize>,
}

/// Graphe de termes minimal avec partage de sous-graphes (DAG).
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
        if out.len() >= max_len || depth > 24 {
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

/// Échantillonne un alphabet d'octets distincts, optionnellement dans une plage donnée
/// pour tester l'abstraction méta-relationnelle sur alphabets disjoints.
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
    let max_len = 36;

    let root = match family {
        CauseFamily::Periodic => {
            let period = 2 + rng.next_usize(4);
            let alpha = sample_alphabet(rng, period, lo, hi);
            let leaves: Vec<usize> = alpha.into_iter().map(|b| tg.add_leaf(b)).collect();
            let motif = tg.add_branch(leaves);
            let reps = (24 / period).max(5);
            tg.add_branch(vec![motif; reps])
        }
        CauseFamily::Symmetric => {
            let half = 4 + rng.next_usize(4);
            let alpha = sample_alphabet(rng, half, lo, hi);
            let fwd: Vec<usize> = alpha.iter().map(|&b| tg.add_leaf(b)).collect();
            let rev: Vec<usize> = fwd.iter().rev().copied().collect();
            let pal = tg.add_branch([fwd, rev].concat());
            tg.add_branch(vec![pal, pal])
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
            tg.add_branch(vec![r, r])
        }
        CauseFamily::SharedMotif => {
            let alpha = sample_alphabet(rng, 7, lo, hi);
            let m0 = tg.add_leaf(alpha[0]);
            let m1 = tg.add_leaf(alpha[1]);
            let m2 = tg.add_leaf(alpha[2]);
            let shared = tg.add_branch(vec![m0, m1, m2, m1]);
            let b0 = tg.add_leaf(alpha[3]);
            let b1 = tg.add_leaf(alpha[4]);
            let b2 = tg.add_leaf(alpha[5]);
            let b3 = tg.add_leaf(alpha[6]);
            tg.add_branch(vec![shared, b0, shared, b1, shared, b2, shared, b3, shared])
        }
        CauseFamily::Arithmetic => {
            let start = lo.wrapping_add((rng.next_usize((hi - lo).max(1) as usize)) as u8);
            let step = (1 + rng.next_usize(7)) as u8;
            let cycle = 4 + rng.next_usize(4);
            let mut children = Vec::with_capacity(28);
            for i in 0..28 {
                let v = start.wrapping_add(step.wrapping_mul((i % cycle) as u8));
                children.push(tg.add_leaf(v));
            }
            tg.add_branch(children)
        }
    };

    let mut effect = tg.linearize(root, max_len);
    while effect.len() < 16 {
        let copy = effect.clone();
        effect.extend_from_slice(&copy);
    }
    effect.truncate(max_len);

    Episode {
        effect,
        hidden_cause: family,
    }
}
