use crate::autodiff::ParamStore;
use crate::model::graph::{GraphParams, ModelConfig};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const PHIS_MAGIC: &[u8; 8] = b"PHISHARP";
const PHIS_VERSION: u32 = 2;

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn write_u32_chk(w: &mut BufWriter<File>, chk: &mut u64, v: u32) -> io::Result<()> {
    let le = v.to_le_bytes();
    *chk = fnv1a64_update(*chk, &le);
    w.write_all(&le)
}

fn write_u64_chk(w: &mut BufWriter<File>, chk: &mut u64, v: u64) -> io::Result<()> {
    let le = v.to_le_bytes();
    *chk = fnv1a64_update(*chk, &le);
    w.write_all(&le)
}

fn write_f32_chk(w: &mut BufWriter<File>, chk: &mut u64, v: f32) -> io::Result<()> {
    let le = v.to_le_bytes();
    *chk = fnv1a64_update(*chk, &le);
    w.write_all(&le)
}

fn read_u32_chk(r: &mut BufReader<File>, chk: &mut u64) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    *chk = fnv1a64_update(*chk, &buf);
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_chk(r: &mut BufReader<File>, chk: &mut u64) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    *chk = fnv1a64_update(*chk, &buf);
    Ok(u64::from_le_bytes(buf))
}

fn read_f32_chk(r: &mut BufReader<File>, chk: &mut u64) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    *chk = fnv1a64_update(*chk, &buf);
    Ok(f32::from_le_bytes(buf))
}

pub fn save_phis(path: impl AsRef<Path>, cfg: &ModelConfig, store: &ParamStore) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    let mut chk: u64 = 0xcbf2_9ce4_8422_2325;

    w.write_all(PHIS_MAGIC)?;
    chk = fnv1a64_update(chk, PHIS_MAGIC);
    write_u32_chk(&mut w, &mut chk, PHIS_VERSION)?;

    // Config
    write_u32_chk(&mut w, &mut chk, cfg.num_nodes as u32)?;
    write_u32_chk(&mut w, &mut chk, cfg.dim as u32)?;
    write_u32_chk(&mut w, &mut chk, cfg.pos_dim as u32)?;
    write_u32_chk(&mut w, &mut chk, cfg.min_steps as u32)?;
    write_u32_chk(&mut w, &mut chk, cfg.max_steps as u32)?;
    write_f32_chk(&mut w, &mut chk, cfg.halt_threshold)?;
    write_u32_chk(&mut w, &mut chk, cfg.max_readout_bytes as u32)?;

    // Paramètres lents Theta
    write_u32_chk(&mut w, &mut chk, store.values.len() as u32)?;
    for i in 0..store.values.len() {
        let name_bytes = store.names[i].as_bytes();
        write_u32_chk(&mut w, &mut chk, name_bytes.len() as u32)?;
        chk = fnv1a64_update(chk, name_bytes);
        w.write_all(name_bytes)?;

        let shape = &store.shapes[i];
        write_u32_chk(&mut w, &mut chk, shape.len() as u32)?;
        for &dim in shape {
            write_u64_chk(&mut w, &mut chk, dim as u64)?;
        }

        let vals = &store.values[i];
        write_u64_chk(&mut w, &mut chk, vals.len() as u64)?;
        for &v in vals {
            write_f32_chk(&mut w, &mut chk, v)?;
        }
    }

    w.write_all(&chk.to_le_bytes())?;
    w.flush()?;
    Ok(())
}

pub fn load_phis(path: impl AsRef<Path>) -> io::Result<(ModelConfig, GraphParams, ParamStore)> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);
    let mut chk: u64 = 0xcbf2_9ce4_8422_2325;

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != PHIS_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "En-tête .phis invalide",
        ));
    }
    chk = fnv1a64_update(chk, &magic);

    let version = read_u32_chk(&mut r, &mut chk)?;
    if version != PHIS_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Version .phis non supportée: {}", version),
        ));
    }

    let cfg = ModelConfig {
        num_nodes: read_u32_chk(&mut r, &mut chk)? as usize,
        dim: read_u32_chk(&mut r, &mut chk)? as usize,
        pos_dim: read_u32_chk(&mut r, &mut chk)? as usize,
        min_steps: read_u32_chk(&mut r, &mut chk)? as usize,
        max_steps: read_u32_chk(&mut r, &mut chk)? as usize,
        halt_threshold: read_f32_chk(&mut r, &mut chk)?,
        max_readout_bytes: read_u32_chk(&mut r, &mut chk)? as usize,
    };

    let num_params = read_u32_chk(&mut r, &mut chk)? as usize;
    let mut store = ParamStore::new();
    for _ in 0..num_params {
        let name_len = read_u32_chk(&mut r, &mut chk)? as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf)?;
        chk = fnv1a64_update(chk, &name_buf);
        let name = String::from_utf8(name_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let ndim = read_u32_chk(&mut r, &mut chk)? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64_chk(&mut r, &mut chk)? as usize);
        }

        let val_len = read_u64_chk(&mut r, &mut chk)? as usize;
        let mut vals = Vec::with_capacity(val_len);
        for _ in 0..val_len {
            vals.push(read_f32_chk(&mut r, &mut chk)?);
        }
        store.add_param(name, vals, shape);
    }

    let mut expected_chk_buf = [0u8; 8];
    r.read_exact(&mut expected_chk_buf)?;
    let expected_chk = u64::from_le_bytes(expected_chk_buf);
    if chk != expected_chk {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Somme de contrôle .phis corrompue",
        ));
    }

    let params = GraphParams::from_store(cfg.clone(), &store)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok((cfg, params, store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::controller::readout_infer;
    use crate::rng::Rng;

    #[test]
    fn phis_roundtrip_preserves_exact_weights_and_inference() {
        let mut rng = Rng::from_seed(101);
        let mut store = ParamStore::new();
        let cfg = ModelConfig {
            num_nodes: 6,
            dim: 8,
            pos_dim: 4,
            min_steps: 2,
            max_steps: 3,
            halt_threshold: 0.9,
            max_readout_bytes: 8,
        };
        let params = GraphParams::init(cfg.clone(), &mut store, &mut rng);
        let sample_in = b"xyxyzz";
        let out_before = readout_infer(&store, &params, sample_in);

        let tmp = std::env::temp_dir().join("phisharp_roundtrip_test.phis");
        save_phis(&tmp, &cfg, &store).expect("save_phis failed");
        let (cfg2, params2, store2) = load_phis(&tmp).expect("load_phis failed");
        let _ = fs::remove_file(&tmp);

        assert_eq!(cfg, cfg2);
        assert_eq!(store.values, store2.values);
        let out_after = readout_infer(&store2, &params2, sample_in);
        assert_eq!(out_before, out_after);
    }
}