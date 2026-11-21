use crate::graph::*;
use std::fs;
use std::path::Path;

pub fn save(g: &Graph, path: &str) -> Result<(), String> {
    if let Some(p) = Path::new(path).parent() {
        if !p.as_os_str().is_empty() { fs::create_dir_all(p).map_err(|x| format!("mkdir: {x}"))?; }
    }
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"PHIS");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&(D as u32).to_le_bytes());
    buf.extend_from_slice(&(N as u32).to_le_bytes());
    buf.extend_from_slice(&(g.edges.len() as u32).to_le_bytes());
    for e in &g.edges {
        buf.extend_from_slice(&e.src.to_le_bytes());
        buf.extend_from_slice(&e.dst.to_le_bytes());
        buf.push(e.kind as u8);
        for v in &e.w { buf.extend_from_slice(&v.to_le_bytes()); }
    }
    for b in &g.bias { for v in b { buf.extend_from_slice(&v.to_le_bytes()); } }
    buf.extend_from_slice(&g.meta.eta_fast.to_le_bytes());
    buf.extend_from_slice(&g.meta.explore_gain.to_le_bytes());
    buf.extend_from_slice(&g.meta.phi_ema.to_le_bytes());
    fs::write(path, buf).map_err(|x| format!("write {path}: {x}"))
}

pub fn load(path: &str) -> Result<Graph, String> {
    let b = fs::read(path).map_err(|x| format!("read {path}: {x}"))?;
    let mut c = Cur { b: &b, p: 0 };
    if c.take(4)? != b"PHIS".as_slice() { return Err("magic PHIS attendu".into()); }
    let ver = c.u32()?; if ver != 1 { return Err(format!("version {ver}")); }
    let d = c.u32()? as usize; let n = c.u32()? as usize;
    if d != D || n != N { return Err("dimensions incompatibles".into()); }
    let ne = c.u32()? as usize;
    let mut edges = Vec::with_capacity(ne);
    for _ in 0..ne {
        let src = c.u32()?; let dst = c.u32()?;
        let kind = EdgeKind::from_code(c.u8()?).ok_or("kind invalide")?;
        let mut w = [0.0; D * D];
        for v in w.iter_mut() { *v = c.f32()?; }
        edges.push(Edge { src, dst, kind, w });
    }
    let mut bias = vec![[0.0; D]; N];
    for bb in bias.iter_mut() { for v in bb.iter_mut() { *v = c.f32()?; } }
    let eta_fast = c.f32()?; let explore_gain = c.f32()?; let phi_ema = c.f32()?;
    let mut g = Graph { edges, incoming: Vec::new(), bias, meta: Meta { eta_fast, explore_gain, phi_ema } };
    g.rebuild_incoming();
    g.assert_non_reciprocal();
    Ok(g)
}

struct Cur<'a> { b: &'a [u8], p: usize }
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() { return Err("fichier tronqué".into()); }
        let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?; Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn f32(&mut self) -> Result<f32, String> {
        let s = self.take(4)?; Ok(f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}