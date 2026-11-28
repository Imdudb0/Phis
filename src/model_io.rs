//! Phisharp V1 — Sérialisation binaire autonome du format `out/sharp.phis`.
//!
//! Structure binaire (little-endian) :
//!   [0..8]   Magic "PHISHARP"
//!   [8..12]  Version de format (u32 = 1)
//!   [12..36] Empreinte d'architecture (6 * u32 : hidden_dim, num_layers,
//!            NODE_FEAT_DIM, GLOBAL_RES_DIM, NUM_TRIPARTITE_EDGE_TYPES, NUM_CONNECT_CATS)
//!   [36..68] Métadonnées d'entraînement (trained_steps: u64, max_difficulty: u8,
//!            reserved: [u8; 7], last_mean_loss: f64, last_mean_log_phi: f64)
//!   [68..76] param_count (u64)
//!   [76.. ]  param_count * f64 (little-endian, tous finis)
//!   [fin-8..] Checksum FNV-1a 64 bits sur tous les octets précédents.

#![forbid(unsafe_code)]

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use crate::backbone::{
    BackboneConfig, GLOBAL_RES_DIM, NODE_FEAT_DIM, NUM_CONNECT_CATS, NUM_TRIPARTITE_EDGE_TYPES,
    RelationalBackbone,
};

pub const PHIS_MAGIC: &[u8; 8] = b"PHISHARP";
pub const PHIS_FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 76;
const CHECKSUM_LEN: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelMetadata {
    pub trained_steps: u64,
    pub max_difficulty: u8,
    pub last_mean_loss: f64,
    pub last_mean_log_phi: f64,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            trained_steps: 0,
            max_difficulty: 0,
            last_mean_loss: 0.0,
            last_mean_log_phi: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoadedModel {
    pub backbone: RelationalBackbone,
    pub metadata: ModelMetadata,
}

#[derive(Debug)]
pub enum ModelIoError {
    Io(io::Error),
    TruncatedFile,
    InvalidMagic,
    UnsupportedVersion(u32),
    ChecksumMismatch { expected: u64, found: u64 },
    ArchitectureMismatch(String),
    NonFiniteParameter(usize),
    InvalidPayloadLength,
    BackboneInit(&'static str),
}

impl fmt::Display for ModelIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelIoError::Io(e) => write!(f, "Erreur E/S : {e}"),
            ModelIoError::TruncatedFile => write!(f, "Fichier .phis tronqué ou vide"),
            ModelIoError::InvalidMagic => write!(f, "En-tête magique PHISHARP invalide"),
            ModelIoError::UnsupportedVersion(v) => {
                write!(f, "Version de format .phis non supportée : {v}")
            }
            ModelIoError::ChecksumMismatch { expected, found } => {
                write!(
                    f,
                    "Checksum .phis corrompu (attendu 0x{expected:016x}, lu 0x{found:016x})"
                )
            }
            ModelIoError::ArchitectureMismatch(msg) => {
                write!(f, "Architecture incompatible : {msg}")
            }
            ModelIoError::NonFiniteParameter(idx) => {
                write!(f, "Paramètre non fini (NaN/Inf) à l'indice {idx}")
            }
            ModelIoError::InvalidPayloadLength => {
                write!(f, "Taille du fichier .phis incohérente avec param_count")
            }
            ModelIoError::BackboneInit(msg) => write!(f, "Initialisation backbone : {msg}"),
        }
    }
}

impl From<io::Error> for ModelIoError {
    fn from(value: io::Error) -> Self {
        ModelIoError::Io(value)
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn read_u32_le(bytes: &[u8], offset: &mut usize) -> Result<u32, ModelIoError> {
    let end = *offset + 4;
    if end > bytes.len() {
        return Err(ModelIoError::TruncatedFile);
    }
    let arr: [u8; 4] = bytes[*offset..end].try_into().unwrap();
    *offset = end;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_le(bytes: &[u8], offset: &mut usize) -> Result<u64, ModelIoError> {
    let end = *offset + 8;
    if end > bytes.len() {
        return Err(ModelIoError::TruncatedFile);
    }
    let arr: [u8; 8] = bytes[*offset..end].try_into().unwrap();
    *offset = end;
    Ok(u64::from_le_bytes(arr))
}

fn read_f64_le(bytes: &[u8], offset: &mut usize) -> Result<f64, ModelIoError> {
    let bits = read_u64_le(bytes, offset)?;
    Ok(f64::from_bits(bits))
}

/// Sérialise un `RelationalBackbone` et ses métadonnées en octets `.phis`.
pub fn encode_phis(
    backbone: &RelationalBackbone,
    metadata: ModelMetadata,
) -> Result<Vec<u8>, ModelIoError> {
    let params = backbone.params();
    for (i, &w) in params.iter().enumerate() {
        if !w.is_finite() {
            return Err(ModelIoError::NonFiniteParameter(i));
        }
    }

    let cfg = backbone.config();
    let total_len = HEADER_LEN + params.len() * 8 + CHECKSUM_LEN;
    let mut buf = Vec::with_capacity(total_len);

    buf.extend_from_slice(PHIS_MAGIC);
    buf.extend_from_slice(&PHIS_FORMAT_VERSION.to_le_bytes());

    buf.extend_from_slice(&(cfg.hidden_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(cfg.num_layers as u32).to_le_bytes());
    buf.extend_from_slice(&(NODE_FEAT_DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(GLOBAL_RES_DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(NUM_TRIPARTITE_EDGE_TYPES as u32).to_le_bytes());
    buf.extend_from_slice(&(NUM_CONNECT_CATS as u32).to_le_bytes());

    buf.extend_from_slice(&metadata.trained_steps.to_le_bytes());
    buf.push(metadata.max_difficulty);
    buf.extend_from_slice(&[0u8; 7]);
    buf.extend_from_slice(&metadata.last_mean_loss.to_bits().to_le_bytes());
    buf.extend_from_slice(&metadata.last_mean_log_phi.to_bits().to_le_bytes());

    buf.extend_from_slice(&(params.len() as u64).to_le_bytes());
    debug_assert_eq!(buf.len(), HEADER_LEN);

    for &w in params {
        buf.extend_from_slice(&w.to_bits().to_le_bytes());
    }

    let checksum = fnv1a_64(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    debug_assert_eq!(buf.len(), total_len);
    Ok(buf)
}

/// Désérialise et vérifie intégralement un tampon `.phis`.
pub fn decode_phis(bytes: &[u8]) -> Result<LoadedModel, ModelIoError> {
    if bytes.len() < HEADER_LEN + CHECKSUM_LEN {
        return Err(ModelIoError::TruncatedFile);
    }
    if &bytes[0..8] != PHIS_MAGIC {
        return Err(ModelIoError::InvalidMagic);
    }

    let payload_end = bytes.len() - CHECKSUM_LEN;
    let mut ck_off = payload_end;
    let found_checksum = read_u64_le(bytes, &mut ck_off)?;
    let expected_checksum = fnv1a_64(&bytes[..payload_end]);
    if found_checksum != expected_checksum {
        return Err(ModelIoError::ChecksumMismatch {
            expected: expected_checksum,
            found: found_checksum,
        });
    }

    let mut off = 8usize;
    let version = read_u32_le(bytes, &mut off)?;
    if version != PHIS_FORMAT_VERSION {
        return Err(ModelIoError::UnsupportedVersion(version));
    }

    let hidden_dim = read_u32_le(bytes, &mut off)? as usize;
    let num_layers = read_u32_le(bytes, &mut off)? as usize;
    let node_feat_dim = read_u32_le(bytes, &mut off)? as usize;
    let global_res_dim = read_u32_le(bytes, &mut off)? as usize;
    let edge_types = read_u32_le(bytes, &mut off)? as usize;
    let conn_cats = read_u32_le(bytes, &mut off)? as usize;

    if node_feat_dim != NODE_FEAT_DIM
        || global_res_dim != GLOBAL_RES_DIM
        || edge_types != NUM_TRIPARTITE_EDGE_TYPES
        || conn_cats != NUM_CONNECT_CATS
    {
        return Err(ModelIoError::ArchitectureMismatch(format!(
            "constantes lues ({node_feat_dim}, {global_res_dim}, {edge_types}, {conn_cats}) != attendues ({NODE_FEAT_DIM}, {GLOBAL_RES_DIM}, {NUM_TRIPARTITE_EDGE_TYPES}, {NUM_CONNECT_CATS})"
        )));
    }

    let trained_steps = read_u64_le(bytes, &mut off)?;
    let max_difficulty = bytes[off];
    off += 8; // 1 octet difficulty + 7 octets réservés
    let last_mean_loss = read_f64_le(bytes, &mut off)?;
    let last_mean_log_phi = read_f64_le(bytes, &mut off)?;
    let param_count = read_u64_le(bytes, &mut off)? as usize;
    debug_assert_eq!(off, HEADER_LEN);

    let expected_payload_bytes = param_count
        .checked_mul(8)
        .ok_or(ModelIoError::InvalidPayloadLength)?;
    if payload_end - off != expected_payload_bytes {
        return Err(ModelIoError::InvalidPayloadLength);
    }

    let mut params = Vec::with_capacity(param_count);
    for idx in 0..param_count {
        let w = read_f64_le(bytes, &mut off)?;
        if !w.is_finite() {
            return Err(ModelIoError::NonFiniteParameter(idx));
        }
        params.push(w);
    }

    let cfg = BackboneConfig {
        hidden_dim,
        num_layers,
    };
    let backbone =
        RelationalBackbone::from_params(cfg, params).map_err(ModelIoError::BackboneInit)?;

    Ok(LoadedModel {
        backbone,
        metadata: ModelMetadata {
            trained_steps,
            max_difficulty,
            last_mean_loss,
            last_mean_log_phi,
        },
    })
}

/// Écrit le fichier `.phis` sur disque en créant les répertoires parents si nécessaire.
pub fn save_phis(
    path: &Path,
    backbone: &RelationalBackbone,
    metadata: ModelMetadata,
) -> Result<usize, ModelIoError> {
    let encoded = encode_phis(backbone, metadata)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, &encoded)?;
    Ok(encoded.len())
}

/// Charge et vérifie un fichier `.phis` depuis le disque.
pub fn load_phis(path: &Path) -> Result<LoadedModel, ModelIoError> {
    let bytes = fs::read(path)?;
    decode_phis(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phis_roundtrip_bit_exact_and_disk_io() {
        let cfg = BackboneConfig {
            hidden_dim: 8,
            num_layers: 2,
        };
        let backbone = RelationalBackbone::new(cfg, 0xCAFE_BABE).unwrap();
        let meta = ModelMetadata {
            trained_steps: 420,
            max_difficulty: 3,
            last_mean_loss: 1.234567,
            last_mean_log_phi: -0.875,
        };

        let bytes = encode_phis(&backbone, meta).unwrap();
        let decoded = decode_phis(&bytes).unwrap();
        assert_eq!(decoded.backbone.config(), cfg);
        assert_eq!(decoded.backbone.params(), backbone.params());
        assert_eq!(decoded.metadata, meta);

        // Aller-retour disque dans un fichier temporaire.
        let mut tmp_path = std::env::temp_dir();
        tmp_path.push("phisharp_test_roundtrip.phis");
        let written = save_phis(&tmp_path, &backbone, meta).unwrap();
        assert_eq!(written, bytes.len());
        let from_disk = load_phis(&tmp_path).unwrap();
        assert_eq!(from_disk.backbone.params(), backbone.params());
        assert_eq!(from_disk.metadata, meta);
        let _ = fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_phis_rejects_corruption_bad_magic_and_nan() {
        let mut backbone = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: 4,
                num_layers: 1,
            },
            7,
        )
        .unwrap();
        let meta = ModelMetadata::default();

        let mut bytes = encode_phis(&backbone, meta).unwrap();

        // 1. Corruption d'un octet dans la charge utile -> échec checksum.
        let mid = HEADER_LEN + 3;
        bytes[mid] ^= 0xFF;
        assert!(matches!(
            decode_phis(&bytes),
            Err(ModelIoError::ChecksumMismatch { .. })
        ));
        bytes[mid] ^= 0xFF;

        // 2. Magic invalide.
        bytes[0] = b'X';
        assert!(matches!(decode_phis(&bytes), Err(ModelIoError::InvalidMagic)));

        // 3. Rejet d'un poids NaN à la sérialisation.
        backbone.params_mut()[0] = f64::NAN;
        assert!(matches!(
            encode_phis(&backbone, meta),
            Err(ModelIoError::NonFiniteParameter(0))
        ));
    }
}
