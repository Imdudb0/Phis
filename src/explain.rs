//! Génération NL : à partir du graphe d'abduction, on verbalise en
//! parcourant les nœuds et en émettant leurs labels (Termes = texte
//! du dictionnaire, Bytes = octets, Concepts = terme le plus proche
//! via c2t du modèle).

use crate::graph::{Graph, NodeKind, D};
use crate::model::Model;
use std::collections::HashSet;

pub fn verbalize(model: &Model, hyp: &Graph, max_items: usize) -> String {
    let mut parts = vec![];
    let mut seen = HashSet::new();
    for node in &hyp.nodes {
        if parts.len() >= max_items { break; }
        match node.kind {
            NodeKind::Term => {
                let ti = node.label as usize;
                if ti < model.term_dict.len() && seen.insert(("t", ti)) {
                    let s = String::from_utf8_lossy(&model.term_dict[ti]).to_string();
                    if !s.is_empty() { parts.push(s); }
                }
            }
            NodeKind::Byte => {
                let b = node.label as u8;
                if b.is_ascii_graphic() || b == b' ' {
                    let s = String::from_utf8_lossy(&[b]).to_string();
                    if seen.insert(("b", b as usize)) { parts.push(s); }
                }
            }
            NodeKind::Concept => {
                let ci = node.label as usize;
                // verbalisation par c2t si dispo, sinon par similarité
                let term = verbalize_concept(model, ci, node.emb);
                if let Some(t) = term {
                    if seen.insert(("c", ci)) { parts.push(t); }
                }
            }
            NodeKind::Var => {}
        }
    }
    if parts.is_empty() { "(no explanation found)".into() } else { parts.join(" ") }
}

fn verbalize_concept(model: &Model, ci: usize, emb: [f32; D]) -> Option<String> {
    // Priorité 1 : c2t appris (par src=ci)
    for (s, t, m) in &model.c2t {
        if *s as usize == ci {
            let ti = *t as usize;
            if ti < model.term_dict.len() {
                return Some(String::from_utf8_lossy(&model.term_dict[ti]).to_string());
            }
        }
    }
    // Priorité 2 : similarité cos sur term_emb
    if model.term_emb.is_empty() { return None; }
    let mut best_i = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, te) in model.term_emb.iter().enumerate() {
        let mut d = 0.0;
        for k in 0..D { d += emb[k] * te[k]; }
        if d > best_dot { best_dot = d; best_i = i; }
    }
    Some(String::from_utf8_lossy(&model.term_dict[best_i]).to_string())
}

pub fn residue(model: &Model, hyp: &Graph, _phi: f64) -> String {
    // Résidu causal : concepts non verbalisés + nœuds de E peu couverts.
    let mut unverbalized = 0usize;
    let mut n_concepts = 0usize;
    for node in &hyp.nodes {
        if node.kind == NodeKind::Concept {
            n_concepts += 1;
            let ci = node.label as usize;
            if verbalize_concept(model, ci, node.emb).is_none() {
                unverbalized += 1;
            }
        }
    }
    format!(
        "concepts:{} unverbalized:{} (a clearer explanation requires more concept-to-term edges)",
        n_concepts, unverbalized
    )
}