//! Artefact `out/sharp.phis`, empreinte de version, gel, et registre des jeux
//! d'évaluation consommés. Zéro dépendance.
//!
//! TRAÇABILITÉ : la déclaration de fermeture est écrite DANS l'artefact et
//! hashée. Un artefact non fermé ne peut pas porter l'étiquette de l'expérience
//! fermée : tout ajout (verbalisation, définitions, ...) est une NOUVELLE
//! expérience, et il ne corrige pas rétroactivement le verdict de la version
//! testée.
//!
//! REGISTRE : un jeu d'évaluation, surtout en langue naturelle, n'est vu qu'une
//! fois par empreinte de modèle. Le registre est consulté par le harness.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::ad::Params;
use crate::eff::ViewCfg;
use crate::emit::EmitCfg;
use crate::abduction::SearchCfg;
use crate::phi::PhiCfg;
use crate::rng::fnv1a;

pub const MAGIC: &[u8; 8] = b"PHISharp";
pub const FORMAT: u32 = 1;
pub const CLOSED_EXPERIMENT: &str = "phisharp-closed-v1";

#[derive(Clone, Copy, Debug)]
pub struct Closure {
    pub ptgrs: bool,
    pub phi: bool,
    pub nl_text: bool,
    pub definitions: bool,
    pub self_described_e: bool,
    pub rendered_h: bool,
    pub intentions: bool,
    pub pretrained_prior: bool,
    pub nl_head: bool,
}

impl Closure {
    pub fn closed() -> Self {
        Closure {
            ptgrs: true,
            phi: true,
            nl_text: false,
            definitions: false,
            self_described_e: false,
            rendered_h: false,
            intentions: false,
            pretrained_prior: false,
            nl_head: false,
        }
    }

    pub fn is_closed(&self) -> bool {
        self.ptgrs
            && self.phi
            && !self.nl_text
            && !self.definitions
            && !self.self_described_e
            && !self.rendered_h
            && !self.intentions
            && !self.pretrained_prior
            && !self.nl_head
    }

    pub fn bits(&self) -> u32 {
        let f = [
            self.ptgrs,
            self.phi,
            self.nl_text,
            self.definitions,
            self.self_described_e,
            self.rendered_h,
            self.intentions,
            self.pretrained_prior,
            self.nl_head,
        ];
        let mut x = 0u32;
        for (i, b) in f.iter().enumerate() {
            if *b {
                x |= 1 << i;
            }
        }
        x
    }

    pub fn from_bits(x: u32) -> Self {
        let g = |i: usize| (x >> i) & 1 == 1;
        Closure {
            ptgrs: g(0),
            phi: g(1),
            nl_text: g(2),
            definitions: g(3),
            self_described_e: g(4),
            rendered_h: g(5),
            intentions: g(6),
            pretrained_prior: g(7),
            nl_head: g(8),
        }
    }
}

#[derive(Clone)]
pub struct Model {
    pub params: Params,
    pub view: ViewCfg,
    pub search: SearchCfg,
    pub emit: EmitCfg,
    pub seed: u64,
    pub episodes: u64,
    pub steps: u64,
    pub closure: Closure,
    pub experiment: String,
    pub fingerprint: u64,
}

fn pu64(b: &mut Vec<u8>, x: u64) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn pu32(b: &mut Vec<u8>, x: u32) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn pf32(b: &mut Vec<u8>, x: f32) {
    b.extend_from_slice(&x.to_le_bytes());
}
fn pu8(b: &mut Vec<u8>, x: u8) {
    b.push(x);
}
fn pstr(b: &mut Vec<u8>, s: &str) {
    pu64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.i.checked_add(n).map(|e| e > self.b.len()).unwrap_or(true) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "artefact tronque"));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u64(&mut self) -> io::Result<u64> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    fn u32(&mut self) -> io::Result<u32> {
        let s = self.take(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(s);
        Ok(u32::from_le_bytes(a))
    }
    fn f32(&mut self) -> io::Result<f32> {
        let s = self.take(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(s);
        Ok(f32::from_le_bytes(a))
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn boolean(&mut self) -> io::Result<bool> {
        Ok(self.u8()? != 0)
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        if n > 4096 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "etiquette trop longue"));
        }
        let s = self.take(n)?;
        String::from_utf8(s.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "etiquette non utf8"))
    }
}

pub fn params_fingerprint(p: &Params) -> u64 {
    let mut b = Vec::new();
    pu64(&mut b, p.nblocks() as u64);
    for (i, d) in p.data.iter().enumerate() {
        pu64(&mut b, p.shapes[i].r as u64);
        pu64(&mut b, p.shapes[i].c as u64);
        for x in d.iter() {
            pf32(&mut b, *x);
        }
    }
    fnv1a(&b)
}

impl Model {
    fn payload(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        pu32(&mut b, FORMAT);
        pu32(&mut b, self.closure.bits());
        pstr(&mut b, &self.experiment);
        pu64(&mut b, self.seed);
        pu64(&mut b, self.episodes);
        pu64(&mut b, self.steps);

        pu64(&mut b, self.view.max_atoms as u64);
        for x in self.view.fanout.iter() {
            pu64(&mut b, *x as u64);
        }
        pu64(&mut b, self.view.topk as u64);
        pu64(&mut b, self.view.eq_cap as u64);
        for x in self.view.ord_dists.iter() {
            pu64(&mut b, *x as u64);
        }

        pu64(&mut b, self.search.max_ops as u64);
        pu64(&mut b, self.search.max_rel);
        pu64(&mut b, self.search.rounds as u64);
        pu8(&mut b, self.search.allow_refutation as u8);
        pu8(&mut b, self.search.explore as u8);
        pf32(&mut b, self.search.lambda_cov);
        pf32(&mut b, self.search.mu_res);
        pu64(&mut b, self.search.phi.mech as u64);
        pu64(&mut b, self.search.phi.purview as u64);
        pu64(&mut b, self.search.phi.mip as u64);
        pu8(&mut b, self.search.phi.perturb as u8);

        pf32(&mut b, self.emit.phi_min);
        pf32(&mut b, self.emit.stab_min);
        pf32(&mut b, self.emit.flow_rho);
        pu64(&mut b, self.emit.depth as u64);
        pu8(&mut b, self.emit.delim);
        pu64(&mut b, self.emit.max_atoms as u64);

        pu64(&mut b, self.params.nblocks() as u64);
        for (i, d) in self.params.data.iter().enumerate() {
            pu64(&mut b, self.params.shapes[i].r as u64);
            pu64(&mut b, self.params.shapes[i].c as u64);
            for x in d.iter() {
                pf32(&mut b, *x);
            }
        }
        b
    }

    pub fn save(&self, path: &str) -> io::Result<u64> {
        if self.experiment == CLOSED_EXPERIMENT {
            assert!(
                self.closure.is_closed(),
                "fermeture violee : cet ajout est une NOUVELLE experience, pas un correctif"
            );
        }
        let mut b = self.payload();
        let fp = fnv1a(&b);
        pu64(&mut b, fp);
        if let Some(par) = Path::new(path).parent() {
            if !par.as_os_str().is_empty() {
                fs::create_dir_all(par)?;
            }
        }
        let mut f = fs::File::create(path)?;
        f.write_all(&b)?;
        f.flush()?;
        Ok(fp)
    }

    pub fn load(path: &str) -> io::Result<Model> {
        let buf = fs::read(path)?;
        if buf.len() < 20 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "artefact trop court"));
        }
        let (payload, tail) = buf.split_at(buf.len() - 8);
        let mut a = [0u8; 8];
        a.copy_from_slice(tail);
        let stored = u64::from_le_bytes(a);
        if fnv1a(payload) != stored {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empreinte de version invalide : artefact altere",
            ));
        }
        let mut r = Rd { b: payload, i: 0 };
        if r.take(8)? != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "magic"));
        }
        if r.u32()? != FORMAT {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "format"));
        }
        let closure = Closure::from_bits(r.u32()?);
        let experiment = r.string()?;
        let seed = r.u64()?;
        let episodes = r.u64()?;
        let steps = r.u64()?;

        let max_atoms = r.u64()? as usize;
        let fanout = [r.u64()? as usize, r.u64()? as usize, r.u64()? as usize];
        let topk = r.u64()? as usize;
        let eq_cap = r.u64()? as usize;
        let ord_dists = [r.u64()? as usize, r.u64()? as usize, r.u64()? as usize];
        let view = ViewCfg { max_atoms, fanout, topk, eq_cap, ord_dists };

        let max_ops = r.u64()? as usize;
        let max_rel = r.u64()?;
        let rounds = r.u64()? as usize;
        let allow_refutation = r.boolean()?;
        let explore = r.boolean()?;
        let lambda_cov = r.f32()?;
        let mu_res = r.f32()?;
        let mech = r.u64()? as usize;
        let purview = r.u64()? as usize;
        let mip = r.u64()? as usize;
        let perturb = r.boolean()?;
        let search = SearchCfg {
            max_ops,
            max_rel,
            rounds,
            allow_refutation,
            explore,
            phi: PhiCfg { mech, purview, mip, perturb },
            lambda_cov,
            mu_res,
        };

        let phi_min = r.f32()?;
        let stab_min = r.f32()?;
        let flow_rho = r.f32()?;
        let depth = r.u64()? as usize;
        let delim = r.u8()?;
        let emax = r.u64()? as usize;
        let emit = EmitCfg { phi_min, stab_min, flow_rho, depth, delim, max_atoms: emax };

        let nb = r.u64()? as usize;
        if nb > 4096 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "trop de blocs"));
        }
        let mut params = Params::default();
        for _ in 0..nb {
            let rr = r.u64()? as usize;
            let cc = r.u64()? as usize;
            let n = rr.saturating_mul(cc);
            if n > 4_000_000 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bloc trop grand"));
            }
            let mut d = Vec::with_capacity(n);
            for _ in 0..n {
                d.push(r.f32()?);
            }
            params.block(rr, cc, &d);
        }

        Ok(Model {
            params,
            view,
            search,
            emit,
            seed,
            episodes,
            steps,
            closure,
            experiment,
            fingerprint: stored,
        })
    }
}

/// Registre des jeux consommés : `(empreinte, identifiant de jeu)`.
/// Un jeu vu deux fois par la même version figée invalide la lecture.
pub struct Ledger {
    pub path: String,
    pub rows: Vec<(u64, String)>,
}

impl Ledger {
    pub fn load(path: &str) -> Ledger {
        let mut rows = Vec::new();
        if let Ok(s) = fs::read_to_string(path) {
            for line in s.lines() {
                let mut it = line.split('\t');
                if let (Some(a), Some(b)) = (it.next(), it.next()) {
                    if let Ok(fp) = u64::from_str_radix(a, 16) {
                        rows.push((fp, b.to_string()));
                    }
                }
            }
        }
        Ledger { path: path.to_string(), rows }
    }

    pub fn seen(&self, fp: u64, set: &str) -> bool {
        self.rows.iter().any(|&(f, ref s)| f == fp && s == set)
    }

    pub fn record(&mut self, fp: u64, set: &str) -> io::Result<()> {
        if self.seen(fp, set) {
            return Ok(());
        }
        if let Some(par) = Path::new(&self.path).parent() {
            if !par.as_os_str().is_empty() {
                fs::create_dir_all(par)?;
            }
        }
        let mut f = fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{:016x}\t{}", fp, set)?;
        self.rows.push((fp, set.to_string()));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::substrate;

    fn tmp(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("phis_{}_{}.phis", std::process::id(), tag));
        p.to_string_lossy().to_string()
    }

    fn model() -> Model {
        let mut rng = Rng::new(19);
        Model {
            params: substrate::init(&mut rng),
            view: ViewCfg::default(),
            search: SearchCfg::default(),
            emit: EmitCfg::default(),
            seed: 19,
            episodes: 3,
            steps: 1,
            closure: Closure::closed(),
            experiment: CLOSED_EXPERIMENT.to_string(),
            fingerprint: 0,
        }
    }

    #[test]
    fn roundtrip_et_empreinte() {
        let m = model();
        let path = tmp("rt");
        let fp = m.save(&path).unwrap();
        let l = Model::load(&path).unwrap();
        assert_eq!(l.fingerprint, fp);
        assert_eq!(l.params.nblocks(), m.params.nblocks());
        assert_eq!(l.params.data[0], m.params.data[0]);
        assert_eq!(l.emit.delim, m.emit.delim);
        assert!(l.closure.is_closed());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn alteration_detectee() {
        let m = model();
        let path = tmp("tamper");
        m.save(&path).unwrap();
        let mut b = fs::read(&path).unwrap();
        let n = b.len();
        b[n / 2] ^= 0x40;
        fs::write(&path, &b).unwrap();
        assert!(Model::load(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[should_panic(expected = "NOUVELLE experience")]
    fn ajout_interdit_sous_etiquette_fermee() {
        let mut m = model();
        m.closure.nl_head = true;
        let path = tmp("closure");
        let _ = m.save(&path);
    }
}