//! Émission de l'explication maximale. AUCUNE tête de sortie entraînée au NL.
//!
//! L'explication est la clôture d'effet du mécanisme intégré : on part des nœuds
//! du mécanisme (ceux que φ a retenus), on suit à l'envers les arêtes que la
//! distribution de pointeurs sélectionne, on accumule un flux causal, et on rend
//! les spans de E au-dessus d'un plancher d'admissibilité. L'alphabet vient
//! donc toujours de l'entrée (« auto-spécifié »).
//!
//! Admissibilité dure : pas de sens trouvé -> pas d'explication. Sortie vide.
//! Le délimiteur est une constante de rendu pré-enregistrée, identique pour
//! les baselines ; elle ne porte aucune information linguistique.
//!
//! Le résidu n'est PAS émis sur stdout : il est lu dans la dynamique, par le
//! harness, via le fichier de diagnostics. B se lit sur stdout, sans accès à A.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs;
use std::io;

use crate::abduction::{self, Outcome, SearchCfg};
use crate::ad::{Params, Tape};
use crate::eff::EffView;
use crate::phi;
use crate::substrate::{self, Fwd, Wiring, P};

#[derive(Clone, Copy, Debug)]
pub struct EmitCfg {
    pub phi_min: f32,
    pub stab_min: f32,
    pub flow_rho: f32,
    pub depth: usize,
    pub delim: u8,
    /// 0 = pas de troncature : l'explication est maximale.
    pub max_atoms: usize,
}

impl Default for EmitCfg {
    fn default() -> Self {
        EmitCfg {
            phi_min: 0.02,
            stab_min: 0.10,
            flow_rho: 0.05,
            depth: 4,
            delim: 0x0A,
            max_atoms: 0,
        }
    }
}

pub struct Explanation {
    pub bytes: Vec<u8>,
    pub atoms: Vec<usize>,
    pub flow: Vec<f32>,
    pub admissible: bool,
    pub coverage: f32,
    pub residue: Vec<(usize, f32)>,
    pub phi: f32,
    pub stab: f32,
}

/// Rendu par pointeurs, sans gabarit. Utilisé aussi par les baselines.
pub fn render(bytes: &[u8], v: &EffView, atoms: &[usize], delim: u8) -> Vec<u8> {
    let mut o = Vec::new();
    for (i, &a) in atoms.iter().enumerate() {
        if i > 0 {
            o.push(delim);
        }
        if a < v.n_atoms {
            let s = a * v.atom_len;
            let e = ((a + 1) * v.atom_len).min(bytes.len());
            if e > s {
                o.extend_from_slice(&bytes[s..e]);
            }
        }
    }
    o
}

fn chosen_sources(t: &Tape, f: &Fwd, w: &Wiring, v: &EffView) -> Vec<Option<(usize, f32)>> {
    let mut out = vec![None; w.na];
    let row = P + 1;
    for a in 0..w.na {
        let mut bk = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for k in 0..row {
            let x = t.at(f.ptr, a * row + k);
            if x > bv {
                bv = x;
                bk = k;
            }
        }
        if bk == 0 {
            continue;
        }
        let eid = w.ptr_eid[a * P + (bk - 1)];
        if eid == usize::MAX {
            continue;
        }
        out[a] = Some((v.edges[eid].src as usize, bv));
    }
    out
}

/// Flux causal : ce que le mécanisme explique, transitivement, borné en
/// profondeur. Relaxation max-chemin, ordre FIFO déterministe, garde bornée.
fn closure_flow(
    t: &Tape,
    f: &Fwd,
    w: &Wiring,
    v: &EffView,
    m: &phi::Mech,
    srcs: &[Option<(usize, f32)>],
    depth: usize,
) -> Vec<f32> {
    let n = v.n();
    let mut rev: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for a in 0..w.na {
        if let Some((s, pr)) = srcs[a] {
            if s < n {
                rev[s].push((a, pr));
            }
        }
    }
    let mut flow = vec![0.0f32; n];
    let mut dq: VecDeque<(usize, usize)> = VecDeque::new();
    for &u in m.m.iter() {
        if u < n {
            let g = t.at(f.mech, u).max(1e-6);
            if g > flow[u] {
                flow[u] = g;
            }
            dq.push_back((u, 0));
        }
    }
    let mut guard = 0usize;
    while let Some((u, d)) = dq.pop_front() {
        guard += 1;
        if guard > 4 * w.na + 64 {
            break;
        }
        if d >= depth {
            continue;
        }
        for &(a, pr) in rev[u].iter() {
            let nf = flow[u] * pr;
            if nf > flow[a] + 1e-9 {
                flow[a] = nf;
                dq.push_back((a, d + 1));
            }
        }
    }
    flow.truncate(w.na);
    flow
}

pub fn explain(
    p: &Params,
    bytes: &[u8],
    v: &EffView,
    w: &Wiring,
    scfg: &SearchCfg,
    ecfg: &EmitCfg,
    seed: u64,
) -> (Explanation, Outcome) {
    let out = abduction::search(p, bytes, v, w, scfg, seed, None);

    // Reconstruction exacte de l'état final : même intervention, mêmes poids.
    let mut t = Tape::new();
    let f = substrate::forward(&mut t, p, v, w, &out.itv);
    let m = phi::build_mech(&t, &f, v, w, &scfg.phi);
    let srcs = chosen_sources(&t, &f, w, v);
    let flow = closure_flow(&t, &f, w, v, &m, &srcs, ecfg.depth);

    let maxf = flow.iter().cloned().fold(0.0f32, f32::max);
    let admissible = out.phi >= ecfg.phi_min && out.stab >= ecfg.stab_min;
    let mut atoms: Vec<usize> = Vec::new();
    if admissible && maxf > 0.0 {
        let tau = ecfg.flow_rho * maxf;
        atoms = (0..w.na).filter(|&a| flow[a] >= tau && flow[a] > 0.0).collect();
        atoms.sort_unstable_by(|&a, &b| {
            flow[b].partial_cmp(&flow[a]).unwrap_or(Ordering::Equal).then(a.cmp(&b))
        });
        if ecfg.max_atoms > 0 {
            atoms.truncate(ecfg.max_atoms);
        }
    }
    let rendered = render(bytes, v, &atoms, ecfg.delim);
    let coverage = atoms.len() as f32 / w.na.max(1) as f32;

    let ex = Explanation {
        bytes: rendered,
        atoms,
        flow,
        admissible,
        coverage,
        residue: out.residue.clone(),
        phi: out.phi,
        stab: out.stab,
    };
    (ex, out)
}

/// Diagnostics : observateur seul, jamais réinjecté. Format ligne/TSV, aucune
/// dépendance. C'est par ici que A se lit ; B se lit sur stdout, à l'aveugle.
pub fn write_diag(path: &str, fp: u64, out: &Outcome, ex: &Explanation) -> io::Result<()> {
    let mut s = String::new();
    s.push_str(&format!("meta\tfp\t{}\n", fp));
    s.push_str(&format!("meta\tops\t{}\n", out.ops));
    s.push_str(&format!("meta\trel\t{}\n", out.rel));
    s.push_str(&format!("meta\tphi\t{:.8}\n", out.phi));
    s.push_str(&format!("meta\tbigphi\t{:.8}\n", out.bigphi));
    s.push_str(&format!("meta\tstab\t{:.8}\n", out.stab));
    s.push_str(&format!("meta\tcov\t{:.8}\n", out.cov));
    s.push_str(&format!("meta\tadmissible\t{}\n", ex.admissible as u8));
    s.push_str(&format!("meta\texplanation_coverage\t{:.8}\n", ex.coverage));
    s.push_str(&format!("meta\tnspans\t{}\n", ex.atoms.len()));
    for i in 0..out.trace.len() {
        s.push_str(&format!(
            "trace\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\n",
            out.trace.step[i],
            out.trace.ops[i],
            out.trace.rel[i],
            out.trace.sharp[i],
            out.trace.bigphi[i],
            out.trace.stab[i],
            out.trace.cov[i],
            out.trace.op[i],
            out.trace.site[i],
            out.trace.residue[i]
        ));
    }
    for (j, &a) in ex.atoms.iter().enumerate() {
        s.push_str(&format!("span\t{}\t{}\t{:.8}\n", j, a, ex.flow[a]));
    }
    for &(a, r) in ex.residue.iter() {
        s.push_str(&format!("residue\t{}\t{:.8}\n", a, r));
    }
    for r in out.program.rules.iter() {
        s.push_str(&format!(
            "rule\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.tgt.0, r.tgt.1, r.tgt.2, r.kind, r.rev, r.rank
        ));
    }
    fs::write(path, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eff::{build, ViewCfg};
    use crate::phi::PhiCfg;
    use crate::rng::Rng;

    fn setup() -> (Vec<u8>, EffView, Wiring, Params) {
        let b: Vec<u8> = (0..48).map(|i| ((i * 3) % 5) as u8).collect();
        let v = build(&b, &ViewCfg::default());
        let w = substrate::wire(&v);
        let mut rng = Rng::new(41);
        let p = substrate::init(&mut rng);
        (b, v, w, p)
    }

    fn tiny() -> SearchCfg {
        SearchCfg {
            max_ops: 1,
            max_rel: 20_000_000,
            phi: PhiCfg { mech: 3, purview: 3, mip: 2, perturb: true },
            ..SearchCfg::default()
        }
    }

    /// Porte d'admissibilité : si le seuil est inatteignable, la sortie est VIDE.
    /// Un échec de B doit pouvoir être causé par « pas de sens trouvé ».
    #[test]
    fn porte_admissibilite_vide() {
        let (b, v, w, p) = setup();
        let e = EmitCfg { phi_min: 2.0, ..EmitCfg::default() };
        let (ex, _) = explain(&p, &b, &v, &w, &tiny(), &e, 3);
        assert!(!ex.admissible);
        assert!(ex.bytes.is_empty());
        assert!(ex.atoms.is_empty());
    }

    #[test]
    fn determinisme_emission() {
        let (b, v, w, p) = setup();
        let e = EmitCfg { phi_min: 0.0, stab_min: 0.0, ..EmitCfg::default() };
        let (a, _) = explain(&p, &b, &v, &w, &tiny(), &e, 77);
        let (c, _) = explain(&p, &b, &v, &w, &tiny(), &e, 77);
        assert_eq!(a.bytes, c.bytes);
        assert_eq!(a.atoms, c.atoms);
        assert_eq!(a.phi, c.phi);
    }

    /// Le rendu ne produit que des octets de E (plus le délimiteur) : aucune
    /// tête de sortie ne peut inventer un octet absent de l'entrée.
    #[test]
    fn rendu_alphabet_vient_de_E() {
        let (b, v, w, p) = setup();
        let e = EmitCfg { phi_min: 0.0, stab_min: 0.0, ..EmitCfg::default() };
        let (ex, _) = explain(&p, &b, &v, &w, &tiny(), &e, 5);
        for &x in ex.bytes.iter() {
            assert!(x == e.delim || b.contains(&x), "octet inventé : {}", x);
        }
    }
}