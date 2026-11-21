//! Modèle persistant. Format binaire explicite (pas de serde).

use std::fs;
use std::io::{self, Read, Write};
use crate::graph::D;
use crate::rng::Rng;

pub const MAGIC: &[u8; 8] = b"PHISHP01";

#[derive(Clone)]
pub struct Model {
    pub byte_emb: [[f32; D]; 256],
    pub transition: Box<[[f32; D]; D]>,
    pub term_dict: Vec<Vec<u8>>,          // terme (mot) en octets
    pub term_def: Vec<Vec<u8>>,           // définition en octets
    pub term_emb: Vec<[f32; D]>,
    pub concept_emb: Vec<[f32; D]>,
    // concept → term, matrice non-réciproque (verbalisation)
    pub c2t: Vec<(u32, u32, Box<[[f32; D]; D]>)>,
    pub trained: bool,
}

impl Model {
    pub fn new(rng: &mut Rng) -> Self {
        let mut byte_emb = [[0.0f32; D]; 256];
        for i in 0..256 { for k in 0..D { byte_emb[i][k] = rng.gauss() * 0.1; } }
        let mut transition = Box::new([[0.0f32; D]; D]);
        for i in 0..D { for j in 0..D { transition[i][j] = rng.gauss() * 0.1; } }
        Model {
            byte_emb, transition,
            term_dict: vec![], term_def: vec![], term_emb: vec![],
            concept_emb: vec![], c2t: vec![], trained: false,
        }
    }

    pub fn load_lang(&mut self, lang: &[(String, String)], rng: &mut Rng) {
        for (term, def) in lang {
            let db = def.as_bytes();
            let mut emb = [0.0f32; D];
            if !db.is_empty() {
                for &b in db { for k in 0..D { emb[k] += self.byte_emb[b as usize][k]; } }
                for k in 0..D { emb[k] /= db.len() as f32; }
            }
            for k in 0..D { emb[k] += rng.gauss() * 0.01; }
            self.term_dict.push(term.as_bytes().to_vec());
            self.term_def.push(db.to_vec());
            self.term_emb.push(emb);
        }
    }

    // --- sérialisation binaire ---
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = fs::File::create(path)?;
        f.write_all(MAGIC)?;
        w_u32(&mut f, D as u32)?;
        // byte_emb
        for i in 0..256 { for k in 0..D { w_f32(&mut f, self.byte_emb[i][k])?; } }
        // transition
        for i in 0..D { for j in 0..D { w_f32(&mut f, self.transition[i][j])?; } }
        // terms
        w_u32(&mut f, self.term_dict.len() as u32)?;
        for (d, df) in self.term_dict.iter().zip(self.term_def.iter()) {
            w_u32(&mut f, d.len() as u32)?; f.write_all(d)?;
            w_u32(&mut f, df.len() as u32)?; f.write_all(df)?;
        }
        for e in &self.term_emb { for k in 0..D { w_f32(&mut f, e[k])?; } }
        // concepts
        w_u32(&mut f, self.concept_emb.len() as u32)?;
        for e in &self.concept_emb { for k in 0..D { w_f32(&mut f, e[k])?; } }
        // c2t
        w_u32(&mut f, self.c2t.len() as u32)?;
        for (s, t, m) in &self.c2t {
            w_u32(&mut f, *s)?; w_u32(&mut f, *t)?;
            for i in 0..D { for j in 0..D { w_f32(&mut f, m[i][j])?; } }
        }
        w_u8(&mut f, self.trained as u8)?;
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut buf = Vec::new();
        fs::File::open(path)?.read_to_end(&mut buf)?;
        let mut r = Reader::new(&buf);
        let magic = r.bytes(8)?;
        if magic != MAGIC { return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic")); }
        let d = r.u32()? as usize;
        if d != D { return Err(io::Error::new(io::ErrorKind::InvalidData, "dim mismatch")); }
        let mut byte_emb = [[0.0f32; D]; 256];
        for i in 0..256 { for k in 0..D { byte_emb[i][k] = r.f32()?; } }
        let mut transition = Box::new([[0.0f32; D]; D]);
        for i in 0..D { for j in 0..D { transition[i][j] = r.f32()?; } }
        let nt = r.u32()? as usize;
        let mut term_dict = Vec::with_capacity(nt);
        let mut term_def = Vec::with_capacity(nt);
        for _ in 0..nt {
            let l = r.u32()? as usize; term_dict.push(r.bytes(l)?.to_vec());
            let l = r.u32()? as usize; term_def.push(r.bytes(l)?.to_vec());
        }
        let mut term_emb = Vec::with_capacity(nt);
        for _ in 0..nt {
            let mut e = [0.0f32; D];
            for k in 0..D { e[k] = r.f32()?; }
            term_emb.push(e);
        }
        let nc = r.u32()? as usize;
        let mut concept_emb = Vec::with_capacity(nc);
        for _ in 0..nc {
            let mut e = [0.0f32; D];
            for k in 0..D { e[k] = r.f32()?; }
            concept_emb.push(e);
        }
        let nce = r.u32()? as usize;
        let mut c2t = Vec::with_capacity(nce);
        for _ in 0..nce {
            let s = r.u32()?; let t = r.u32()?;
            let mut m = Box::new([[0.0f32; D]; D]);
            for i in 0..D { for j in 0..D { m[i][j] = r.f32()?; } }
            c2t.push((s, t, m));
        }
        let trained = r.u8()? != 0;
        Ok(Model { byte_emb, transition, term_dict, term_def, term_emb,
                   concept_emb, c2t, trained })
    }
}

fn w_u8<W: Write>(f: &mut W, v: u8) -> io::Result<()> { f.write_all(&[v]) }
fn w_u32<W: Write>(f: &mut W, v: u32) -> io::Result<()> { f.write_all(&v.to_le_bytes()) }
fn w_f32<W: Write>(f: &mut W, v: f32) -> io::Result<()> { f.write_all(&v.to_le_bytes()) }

struct Reader<'a> { b: &'a [u8], p: usize }
impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self { Reader { b, p: 0 } }
    fn bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.bytes(1)?[0]) }
    fn u32(&mut self) -> io::Result<u32> {
        let s = self.bytes(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn f32(&mut self) -> io::Result<f32> {
        let s = self.bytes(4)?;
        Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}