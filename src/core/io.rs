use crate::core::flow::FlowConfig;
use crate::core::phi_sharp::{ModelParams, PhiSharpConfig};
use crate::core::projection::ProjectionConfig;
use crate::core::tensor::Mat;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"PHISHARP";
const FORMAT_VERSION: u32 = 1;

/// Point de sauvegarde complet d'un modèle Phisharp (`out/sharp.phis`).
#[derive(Debug, Clone)]
pub struct PhisCheckpoint {
    pub params: ModelParams,
    pub phi_cfg: PhiSharpConfig,
    pub flow_cfg: FlowConfig,
    pub proj_cfg: ProjectionConfig,
}

impl PhisCheckpoint {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Impossible de créer le dossier {:?}: {}", parent, e))?;
            }
        }

        let payload = self.encode_payload();
        let checksum = fnv1a_64(&payload);

        let mut file = File::create(path_ref)
            .map_err(|e| format!("Impossible de créer {:?}: {}", path_ref, e))?;
        file.write_all(MAGIC).map_err(|e| e.to_string())?;
        file.write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&(payload.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&checksum.to_le_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&payload).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)
            .map_err(|e| format!("Impossible d'ouvrir {:?}: {}", path_ref, e))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;

        if bytes.len() < 28 {
            return Err("Fichier .phis tronqué (en-tête incomplet)".to_string());
        }
        if &bytes[0..8] != MAGIC {
            return Err("En-tête magique invalide (attendu PHISHARP)".to_string());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(format!(
                "Version .phis non supportée: {} (attendue {})",
                version, FORMAT_VERSION
            ));
        }
        let payload_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
        let expected_crc = u64::from_le_bytes(bytes[20..28].try_into().unwrap());

        if bytes.len() != 28 + payload_len {
            return Err("Longueur de charge utile .phis incohérente".to_string());
        }
        let payload = &bytes[28..];
        let actual_crc = fnv1a_64(payload);
        if actual_crc != expected_crc {
            return Err(format!(
                "Checksum FNV-1a invalide (corrompu) : {:016X} != {:016X}",
                actual_crc, expected_crc
            ));
        }

        Self::decode_payload(payload)
    }

    fn encode_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let p = &self.params;

        // Dimensions
        write_u64(&mut buf, p.n as u64);
        write_u64(&mut buf, p.d as u64);
        write_u64(&mut buf, p.h as u64);

        // PhiSharpConfig
        write_f64(&mut buf, self.phi_cfg.alpha);
        write_f64(&mut buf, self.phi_cfg.beta);
        write_f64(&mut buf, self.phi_cfg.lambda_bi);
        write_f64(&mut buf, self.phi_cfg.hinge_tau);
        write_f64(&mut buf, self.phi_cfg.smin_eps);
        write_u64(&mut buf, self.phi_cfg.max_krylov as u64);
        write_u64(&mut buf, self.phi_cfg.max_inv_power as u64);

        // FlowConfig
        write_u64(&mut buf, self.flow_cfg.max_steps as u64);
        write_f64(&mut buf, self.flow_cfg.dt_init);
        write_f64(&mut buf, self.flow_cfg.dt_min);
        write_f64(&mut buf, self.flow_cfg.dt_max);
        write_f64(&mut buf, self.flow_cfg.eta);
        write_f64(&mut buf, self.flow_cfg.omega);
        write_f64(&mut buf, self.flow_cfg.omega_perp);
        write_f64(&mut buf, self.flow_cfg.max_perp_norm);
        write_f64(&mut buf, self.flow_cfg.gamma_abs);
        write_f64(&mut buf, self.flow_cfg.abs_eps);
        write_f64(&mut buf, self.flow_cfg.ball_radius_k);
        write_f64(&mut buf, self.flow_cfg.lyapunov_tol);

        // ProjectionConfig
        write_u64(&mut buf, self.proj_cfg.out_len as u64);
        write_u64(&mut buf, self.proj_cfg.max_steps as u64);
        write_f64(&mut buf, self.proj_cfg.eta_init);
        write_f64(&mut buf, self.proj_cfg.eta_min);
        write_f64(&mut buf, self.proj_cfg.eta_max);
        write_f64(&mut buf, self.proj_cfg.lambda_bi);
        write_f64(&mut buf, self.proj_cfg.gamma_alphabet);
        write_f64(&mut buf, self.proj_cfg.gamma_anticopy);
        write_f64(&mut buf, self.proj_cfg.tau_sharp);
        write_f64(&mut buf, self.proj_cfg.tol_step_l12);

        // ModelParams
        write_vec_f64(&mut buf, &p.w_dir);
        write_vec_f64(&mut buf, &p.w_comp);
        write_vec_f64(&mut buf, &p.b_edge);
        write_vec_f64(&mut buf, &p.w_out_uni);
        write_vec_f64(&mut buf, &p.b_out_uni);
        write_vec_f64(&mut buf, &p.w_out_bi);
        write_vec_f64(&mut buf, &p.b_out_bi);
        write_vec_f64(&mut buf, &p.w_enc_uni);
        write_vec_f64(&mut buf, &p.w_enc_bi);
        write_vec_f64(&mut buf, &p.w_init_asym);
        write_vec_f64(&mut buf, &p.w_sol.data);
        write_vec_f64(&mut buf, &p.nu_sol);

        buf
    }

    fn decode_payload(data: &[u8]) -> Result<Self, String> {
        let mut cursor = 0usize;

        let n = read_u64(data, &mut cursor)? as usize;
        let d = read_u64(data, &mut cursor)? as usize;
        let h = read_u64(data, &mut cursor)? as usize;

        let phi_cfg = PhiSharpConfig {
            alpha: read_f64(data, &mut cursor)?,
            beta: read_f64(data, &mut cursor)?,
            lambda_bi: read_f64(data, &mut cursor)?,
            hinge_tau: read_f64(data, &mut cursor)?,
            smin_eps: read_f64(data, &mut cursor)?,
            max_krylov: read_u64(data, &mut cursor)? as usize,
            max_inv_power: read_u64(data, &mut cursor)? as usize,
        };

        let flow_cfg = FlowConfig {
            max_steps: read_u64(data, &mut cursor)? as usize,
            dt_init: read_f64(data, &mut cursor)?,
            dt_min: read_f64(data, &mut cursor)?,
            dt_max: read_f64(data, &mut cursor)?,
            eta: read_f64(data, &mut cursor)?,
            omega: read_f64(data, &mut cursor)?,
            omega_perp: read_f64(data, &mut cursor)?,
            max_perp_norm: read_f64(data, &mut cursor)?,
            gamma_abs: read_f64(data, &mut cursor)?,
            abs_eps: read_f64(data, &mut cursor)?,
            ball_radius_k: read_f64(data, &mut cursor)?,
            lyapunov_tol: read_f64(data, &mut cursor)?,
        };

        let proj_cfg = ProjectionConfig {
            out_len: read_u64(data, &mut cursor)? as usize,
            max_steps: read_u64(data, &mut cursor)? as usize,
            eta_init: read_f64(data, &mut cursor)?,
            eta_min: read_f64(data, &mut cursor)?,
            eta_max: read_f64(data, &mut cursor)?,
            lambda_bi: read_f64(data, &mut cursor)?,
            gamma_alphabet: read_f64(data, &mut cursor)?,
            gamma_anticopy: read_f64(data, &mut cursor)?,
            tau_sharp: read_f64(data, &mut cursor)?,
            tol_step_l12: read_f64(data, &mut cursor)?,
        };

        let w_dir = read_vec_f64(data, &mut cursor)?;
        let w_comp = read_vec_f64(data, &mut cursor)?;
        let b_edge = read_vec_f64(data, &mut cursor)?;
        let w_out_uni = read_vec_f64(data, &mut cursor)?;
        let b_out_uni = read_vec_f64(data, &mut cursor)?;
        let w_out_bi = read_vec_f64(data, &mut cursor)?;
        let b_out_bi = read_vec_f64(data, &mut cursor)?;
        let w_enc_uni = read_vec_f64(data, &mut cursor)?;
        let w_enc_bi = read_vec_f64(data, &mut cursor)?;
        let w_init_asym = read_vec_f64(data, &mut cursor)?;
        let w_sol_data = read_vec_f64(data, &mut cursor)?;
        let nu_sol = read_vec_f64(data, &mut cursor)?;

        if w_sol_data.len() != n * n {
            return Err("Dimensions incohérentes pour w_sol".to_string());
        }

        let params = ModelParams {
            n,
            d,
            h,
            w_dir,
            w_comp,
            b_edge,
            w_out_uni,
            b_out_uni,
            w_out_bi,
            b_out_bi,
            w_enc_uni,
            w_enc_bi,
            w_init_asym,
            w_sol: Mat {
                n,
                data: w_sol_data,
            },
            nu_sol,
        };

        Ok(Self {
            params,
            phi_cfg,
            flow_cfg,
            proj_cfg,
        })
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_vec_f64(buf: &mut Vec<u8>, slice: &[f64]) {
    write_u64(buf, slice.len() as u64);
    for &v in slice {
        write_f64(buf, v);
    }
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, String> {
    if *cursor + 8 > data.len() {
        return Err("Fin de fichier inattendue (u64)".to_string());
    }
    let v = u64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_f64(data: &[u8], cursor: &mut usize) -> Result<f64, String> {
    if *cursor + 8 > data.len() {
        return Err("Fin de fichier inattendue (f64)".to_string());
    }
    let v = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(v)
}

fn read_vec_f64(data: &[u8], cursor: &mut usize) -> Result<Vec<f64>, String> {
    let len = read_u64(data, cursor)? as usize;
    if *cursor + len * 8 > data.len() {
        return Err("Fin de fichier inattendue (Vec<f64>)".to_string());
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_f64(data, cursor)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ho_ptgrs::rng::Rng;

    #[test]
    fn test_phis_checkpoint_roundtrip_and_corruption_check() {
        let mut rng = Rng::from_seed(1337);
        let ckpt = PhisCheckpoint {
            params: ModelParams::random_init(5, 2, 6, &mut rng),
            phi_cfg: PhiSharpConfig::default(),
            flow_cfg: FlowConfig::default(),
            proj_cfg: ProjectionConfig::default(),
        };

        let path = std::env::temp_dir().join("phisharp_roundtrip_test.phis");
        ckpt.save(&path).expect("Sauvegarde .phis réussie");

        let loaded = PhisCheckpoint::load(&path).expect("Chargement .phis réussi");
        assert_eq!(loaded.params.n, ckpt.params.n);
        assert_eq!(loaded.params.d, ckpt.params.d);
        assert_eq!(loaded.params.h, ckpt.params.h);
        assert_eq!(loaded.params.w_dir, ckpt.params.w_dir);
        assert_eq!(loaded.params.w_sol.data, ckpt.params.w_sol.data);
        assert_eq!(loaded.proj_cfg.out_len, ckpt.proj_cfg.out_len);

        // Corrompre 1 octet et vérifier le rejet par FNV-1a
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, &raw).unwrap();
        assert!(PhisCheckpoint::load(&path).is_err());

        let _ = fs::remove_file(&path);
    }
}
