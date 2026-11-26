use crate::graph::{PhiGraph, RawSubstrate, SEARCH_STEPS};
use crate::phi::{
    DiscoveredRegime, ProjectionMatch, abductive_discontinuity, byte_overlap_ratio, cosine_sim,
    cut_discontinuity, discover_causal_regimes,
};

#[derive(Clone, Debug)]
pub struct ResidualSplit {
    pub parts: Vec<(usize, usize)>,
    pub single_err: f32,
    pub split_err: f32,
    pub cross_cos: f32,
}

#[derive(Clone, Debug)]
pub struct ComposedConcept {
    pub name_bytes: Vec<u8>,
    pub clause_span: (usize, usize),
    pub mold_set: Vec<usize>,
    pub hop_depth: usize,
    pub ref_labels: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct MultiHopMatch {
    pub concept: ComposedConcept,
    pub score: f32,
    pub coverage: f32,
    pub precision: f32,
}

#[derive(Clone, Debug)]
pub struct MultiHopAnalysis {
    pub mold_regimes: Vec<DiscoveredRegime>,
    pub mold_keys: Vec<Option<Vec<u8>>>,
    pub target_spans: Vec<(usize, usize)>,
    pub residual_info: Option<(f32, f32, f32)>,
    pub hop1_per_part: Vec<Vec<ProjectionMatch>>,
    pub hop2_matches: Vec<MultiHopMatch>,
    pub definitions_found: usize,
    pub demoted_meta_count: usize,
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn trim_edge_ws(bytes: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = bytes.len();
    while s < e && bytes[s] <= b' ' {
        s += 1;
    }
    while e > s && bytes[e - 1] <= b' ' {
        e -= 1;
    }
    &bytes[s..e]
}

fn contains_slice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn find_standalone_positions(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > haystack.len() {
        return out;
    }
    let word_like = needle.iter().all(|&b| is_word_byte(b));
    let mut i = 0usize;
    while i + needle.len() <= haystack.len() {
        if &haystack[i..i + needle.len()] == needle {
            let left_ok = !word_like || i == 0 || !is_word_byte(haystack[i - 1]);
            let right_ok = !word_like
                || (i + needle.len() == haystack.len())
                || !is_word_byte(haystack[i + needle.len()]);
            if left_ok && right_ok {
                out.push(i);
                i += needle.len();
                continue;
            }
        }
        i += 1;
    }
    out
}

pub fn decompose_by_residual(graph: &PhiGraph, bytes: &[u8]) -> ResidualSplit {
    let n = bytes.len();
    let sub = RawSubstrate::from_bytes(bytes);
    let rels = graph.encode_relations(&sub);
    let tr0 = graph.abduce(&rels, None);
    let single_err = tr0.final_mean_err;

    if n < 30 {
        return ResidualSplit {
            parts: vec![(0, n)],
            single_err,
            split_err: single_err,
            cross_cos: 1.0,
        };
    }

    let mut pref_err = vec![0.0f32; n + 1];
    for t in 0..n {
        pref_err[t + 1] = pref_err[t] + tr0.final_pos_err[t];
    }

    let mut proposals: Vec<(usize, f32)> = Vec::new();
    let mut p = 14usize;
    while p + 14 <= n {
        let ml = pref_err[p] / (p as f32);
        let mr = (pref_err[n] - pref_err[p]) / ((n - p) as f32);
        let res_jump = (ml - mr).abs();
        let cut_d = cut_discontinuity(&sub, 0, n, p);
        let abd_d = if p % 4 == 0 {
            let l0 = p.saturating_sub(16);
            let r1 = (p + 16).min(n);
            abductive_discontinuity(graph, &rels, (l0, p), (p, r1), 4)
        } else {
            0.0
        };
        proposals.push((p, res_jump + 0.45 * cut_d + 0.55 * abd_d));
        p += 2;
    }
    proposals.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cands: Vec<usize> = Vec::new();
    for (cp, _) in proposals {
        if cands.iter().all(|&x| x.abs_diff(cp) >= 6) {
            cands.push(cp);
            if cands.len() >= 4 {
                break;
            }
        }
    }
    if cands.iter().all(|&x| x.abs_diff(n / 2) >= 6) {
        cands.push(n / 2);
    }

    let eval_cut = |cp: usize| -> (f32, f32, f32, f32) {
        let rl = graph.encode_bytes(&bytes[0..cp]);
        let rr = graph.encode_bytes(&bytes[cp..n]);
        let tl = graph.abduce(&rl, None);
        let tr = graph.abduce(&rr, None);
        let split_e =
            ((cp as f32) * tl.final_mean_err + ((n - cp) as f32) * tr.final_mean_err) / (n as f32);
        let cos_lr = cosine_sim(&tl.final_state, &tr.final_state);
        let (_, _, e_lr) = graph.evaluate_hypothesis(&tl.final_state, &rr);
        let (_, _, e_rl) = graph.evaluate_hypothesis(&tr.final_state, &rl);
        let cross_e = 0.5 * (e_lr + e_rl);
        let cross_gap = ((cross_e - split_e) / (cross_e + 0.08)).clamp(0.0, 1.0);
        let obj = (single_err - split_e) + 0.30 * cross_gap * (1.0 - cos_lr.max(0.0));
        (obj, split_e, cos_lr, cross_gap)
    };

    let mut best_p = n / 2;
    let mut best_obj = f32::NEG_INFINITY;
    let mut best_res = (single_err, 1.0f32, 0.0f32);

    for cp in cands {
        let (obj, se, cos_lr, cg) = eval_cut(cp);
        if obj > best_obj {
            best_obj = obj;
            best_p = cp;
            best_res = (se, cos_lr, cg);
        }
    }

    let lo = best_p.saturating_sub(2).max(12);
    let hi = (best_p + 2).min(n.saturating_sub(12));
    for cp in lo..=hi {
        if cp == best_p {
            continue;
        }
        let (obj, se, cos_lr, cg) = eval_cut(cp);
        if obj > best_obj {
            best_obj = obj;
            best_p = cp;
            best_res = (se, cos_lr, cg);
        }
    }

    let (split_err, cross_cos, cross_gap) = best_res;
    let significant_drop = (single_err - split_err) > 0.025 || split_err < 0.88 * single_err;
    if cross_cos < 0.55 && cross_gap > 0.22 && significant_drop {
        ResidualSplit {
            parts: vec![(0, best_p), (best_p, n)],
            single_err,
            split_err,
            cross_cos,
        }
    } else {
        ResidualSplit {
            parts: vec![(0, n)],
            single_err,
            split_err: single_err,
            cross_cos: 1.0,
        }
    }
}

fn extract_word_tokens(slice: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < slice.len() {
        while i < slice.len() && !is_word_byte(slice[i]) {
            i += 1;
        }
        let start = i;
        while i < slice.len() && is_word_byte(slice[i]) {
            i += 1;
        }
        if i - start >= 3 {
            let tok = slice[start..i].to_vec();
            if !out.contains(&tok) {
                out.push(tok);
            }
        }
    }
    out
}

fn demote_metalanguage_regimes(
    bytes: &[u8],
    raw: &[DiscoveredRegime],
) -> (Vec<DiscoveredRegime>, usize) {
    if raw.len() < 3 {
        return (raw.to_vec(), 0);
    }

    let mut kept: Vec<DiscoveredRegime> = Vec::with_capacity(raw.len());
    let mut kept_tag_tokens: Vec<Vec<Vec<u8>>> = Vec::with_capacity(raw.len());
    let mut demoted = 0usize;

    for (idx, &reg) in raw.iter().enumerate() {
        let prev_end = if kept.is_empty() {
            0
        } else {
            kept.last().unwrap().core_end
        };
        let core_slice = &bytes[reg.core_start..reg.core_end];

        let mut echoed_molds = 0usize;
        let mut echoed_bytes = 0usize;
        if kept.len() >= 2 && idx + 1 < raw.len() {
            for m_i in 0..kept.len() {
                let mut hit_len = 0usize;
                for tok in &kept_tag_tokens[m_i] {
                    let unique_to_mi = (0..kept.len())
                        .all(|m_j| m_j == m_i || !kept_tag_tokens[m_j].contains(tok));
                    if unique_to_mi {
                        let cnt = find_standalone_positions(core_slice, tok).len();
                        if cnt > 0 {
                            hit_len += cnt * tok.len();
                        }
                    }
                }
                if hit_len > 0 {
                    echoed_molds += 1;
                    echoed_bytes += hit_len;
                }
            }
        }

        if echoed_molds >= 2 || (echoed_molds >= 1 && echoed_bytes * 4 >= core_slice.len()) {
            demoted += 1;
            continue;
        }

        let tag_start = prev_end.max(reg.core_start.saturating_sub(28));
        let tag_slice = &bytes[tag_start..reg.core_start];
        kept_tag_tokens.push(extract_word_tokens(tag_slice));
        kept.push(reg);
    }

    let k = kept.len();
    if k == 0 {
        return (raw.to_vec(), 0);
    }
    let lead_gap = kept[0].core_start;
    let trail_gap = bytes.len().saturating_sub(kept[k - 1].core_end);
    let suffix_mode = trail_gap >= 3 && lead_gap < 3;

    for i in 0..k {
        let prev_end = if i == 0 { 0 } else { kept[i - 1].core_end };
        let next_start = if i + 1 == k {
            bytes.len()
        } else {
            kept[i + 1].core_start
        };
        kept[i].referent = if suffix_mode {
            if next_start.saturating_sub(kept[i].core_end) >= 3 {
                Some((kept[i].core_end, next_start))
            } else {
                None
            }
        } else if kept[i].core_start.saturating_sub(prev_end) >= 3 {
            Some((prev_end, kept[i].core_start))
        } else {
            None
        };
    }

    (kept, demoted)
}

fn project_span_to_molds(
    graph: &PhiGraph,
    bytes: &[u8],
    q_span: (usize, usize),
    molds: &[DiscoveredRegime],
) -> Vec<ProjectionMatch> {
    let q_bytes = &bytes[q_span.0..q_span.1];
    let q_rels = graph.encode_bytes(q_bytes);
    let q_trace = graph.abduce(&q_rels, None);

    let mut matches = Vec::with_capacity(molds.len());
    for (idx, reg) in molds.iter().enumerate() {
        let core_bytes = &bytes[reg.core_start..reg.core_end];
        let core_rels = graph.encode_bytes(core_bytes);
        let core_trace = graph.abduce(&core_rels, None);

        let (_, _, fwd_err) = graph.evaluate_hypothesis(&q_trace.final_state, &core_rels);
        let (_, _, bwd_err) = graph.evaluate_hypothesis(&core_trace.final_state, &q_rels);
        let cross_err = 0.5 * (fwd_err + bwd_err);

        let cross_gen = (-2.2 * cross_err).exp().clamp(0.0, 1.0);
        let cos_sim = cosine_sim(&q_trace.final_state, &core_trace.final_state);
        let cos_pos = 0.5 * (1.0 + cos_sim);
        let score = (cross_gen * cos_pos).sqrt();

        let byte_overlap = byte_overlap_ratio(q_bytes, core_bytes);
        matches.push(ProjectionMatch {
            dst_idx: idx,
            score,
            cross_err,
            cos_sim,
            core_start: reg.core_start,
            core_end: reg.core_end,
            referent: reg.referent,
            byte_overlap,
        });
    }
    matches.sort_by(|a, b| b.score.total_cmp(&a.score));
    matches
}

fn find_discriminative_key(
    bytes: &[u8],
    tag_spans: &[(usize, usize)],
    mold_idx: usize,
    def_zones: &[(usize, usize)],
) -> Option<Vec<u8>> {
    let (ts, te) = tag_spans[mold_idx];
    let tag = &bytes[ts..te];
    if tag.len() < 3 || def_zones.is_empty() {
        return None;
    }

    let mut best: Option<(usize, usize, Vec<u8>)> = None;
    for s in 0..tag.len() {
        if tag[s] <= b' ' {
            continue;
        }
        for e in (s + 3)..=tag.len() {
            if tag[e - 1] <= b' ' {
                continue;
            }
            let cand = &tag[s..e];
            if cand.iter().any(|&b| b < 32) {
                continue;
            }
            let unique = tag_spans.iter().enumerate().all(|(j, &(os, oe))| {
                j == mold_idx || !contains_slice(&bytes[os..oe], cand)
            });
            if !unique {
                continue;
            }
            let mut hits = 0usize;
            for &(ds, de) in def_zones {
                hits += find_standalone_positions(&bytes[ds..de], cand).len();
            }
            if hits == 0 {
                continue;
            }
            let better = match &best {
                None => true,
                Some((blen, bhits, _)) => {
                    cand.len() > *blen || (cand.len() == *blen && hits > *bhits)
                }
            };
            if better {
                best = Some((cand.len(), hits, cand.to_vec()));
            }
        }
    }
    best.map(|(_, _, k)| k)
}

#[derive(Clone, Debug)]
struct KeyEntry {
    key: Vec<u8>,
    molds: Vec<usize>,
    hop: usize,
    defined_in: Option<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct KeyHit {
    pos: usize,
    end: usize,
    entry_idx: usize,
}

/// Nettoie les affixes boilerplate purement non-alphanumériques (`=`, `:`, `->`, `|`, `;`, espaces)
/// sans jamais rogner une lettre/chiffre d'un identifiant de concept.
fn strip_common_head_affixes(raw_heads: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut cleaned: Vec<Vec<u8>> = raw_heads
        .iter()
        .map(|h| {
            let slice = match h.iter().rposition(|&b| b == b'\n') {
                Some(nl) if trim_edge_ws(&h[nl + 1..]).len() >= 2 => &h[nl + 1..],
                _ => *h,
            };
            trim_edge_ws(slice).to_vec()
        })
        .collect();

    if cleaned.len() == 1 {
        let h = &mut cleaned[0];
        while h.len() > 2 && !is_word_byte(*h.last().unwrap()) {
            h.pop();
        }
        *h = trim_edge_ws(h).to_vec();
        return cleaned;
    }

    // 1. Suffixe boilerplate commun non-alphanumérique (ex: `" ="`, `":"`, `" ->"`)
    let min_len = cleaned.iter().map(|h| h.len()).min().unwrap_or(0);
    let mut suf = 0usize;
    while suf + 2 < min_len {
        let b = cleaned[0][cleaned[0].len() - 1 - suf];
        if !is_word_byte(b) && cleaned.iter().all(|h| h[h.len() - 1 - suf] == b) {
            suf += 1;
        } else {
            break;
        }
    }
    if suf > 0 {
        for h in &mut cleaned {
            let new_len = h.len().saturating_sub(suf);
            h.truncate(new_len);
            *h = trim_edge_ws(h).to_vec();
        }
    }

    // 2. Préfixe séparateur commun aux en-têtes 1..k (ex: `"| "`, `"; "`), non-alphanumérique uniquement
    let tail_min = cleaned[1..].iter().map(|h| h.len()).min().unwrap_or(0);
    let mut pre = 0usize;
    while pre + 2 < tail_min {
        let b = cleaned[1][pre];
        if !is_word_byte(b) && cleaned[1..].iter().all(|h| h[pre] == b) {
            pre += 1;
        } else {
            break;
        }
    }
    if pre > 0 {
        let sep_pre = cleaned[1][0..pre].to_vec();
        // Si cleaned[0] contient ce séparateur en flux plat (après une queue résiduelle), coupe après
        if let Some(pos) = cleaned[0]
            .windows(sep_pre.len())
            .rposition(|w| w == sep_pre.as_slice())
        {
            let tail = trim_edge_ws(&cleaned[0][pos + sep_pre.len()..]);
            if tail.len() >= 2 {
                cleaned[0] = tail.to_vec();
            }
        }
        for h in &mut cleaned[1..] {
            *h = trim_edge_ws(&h[pre..]).to_vec();
        }
    }

    // 3. Préfixe résiduel non-alphanumérique commun à tous (ex: `"- "`, `"| "`)
    let all_min = cleaned.iter().map(|h| h.len()).min().unwrap_or(0);
    let mut ap = 0usize;
    while ap + 2 < all_min {
        let b = cleaned[0][ap];
        if !is_word_byte(b) && cleaned.iter().all(|h| h[ap] == b) {
            ap += 1;
        } else {
            break;
        }
    }
    if ap > 0 {
        for h in &mut cleaned {
            *h = trim_edge_ws(&h[ap..]).to_vec();
        }
    }

    cleaned
}

fn extract_definitions_in_zone(
    zone_bytes: &[u8],
    zone_offset: usize,
    entries: &[KeyEntry],
) -> Vec<ComposedConcept> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| entries[b].key.len().cmp(&entries[a].key.len()));

    let mut used = vec![false; zone_bytes.len()];
    let mut hits: Vec<KeyHit> = Vec::new();
    for ei in order {
        let key = &entries[ei].key;
        let def_in = entries[ei].defined_in;
        for pos in find_standalone_positions(zone_bytes, key) {
            let end = pos + key.len();
            let abs_pos = zone_offset + pos;
            let abs_end = zone_offset + end;
            // Évite l'auto-référence d'un concept Hop-2 à l'intérieur de sa propre clause
            if let Some((ds, de)) = def_in
                && abs_pos >= ds
                && abs_end <= de
            {
                continue;
            }
            if used[pos..end].iter().any(|&u| u) {
                continue;
            }
            for u in &mut used[pos..end] {
                *u = true;
            }
            hits.push(KeyHit {
                pos,
                end,
                entry_idx: ei,
            });
        }
    }
    hits.sort_by_key(|h| h.pos);
    if hits.len() < 2 {
        return Vec::new();
    }

    let mut gaps: Vec<&[u8]> = Vec::with_capacity(hits.len() - 1);
    for w in hits.windows(2) {
        gaps.push(&zone_bytes[w[0].end..w[1].pos]);
    }
    let min_gap = gaps.iter().map(|g| g.len()).min().unwrap_or(0);

    let mut dom_conn: Option<&[u8]> = None;
    let mut dom_count = 1usize;
    for &g in &gaps {
        if g.len() <= 8 && !g.contains(&b'\n') {
            let cnt = gaps.iter().filter(|&&other| other == g).count();
            if cnt > dom_count
                || (cnt == dom_count && cnt >= 2 && dom_conn.is_some_and(|d| g.len() < d.len()))
            {
                dom_count = cnt;
                dom_conn = Some(g);
            }
        }
    }

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![0usize];
    for g_idx in 0..gaps.len() {
        let gap = gaps[g_idx];
        let h_prev = &hits[g_idx];
        let h_next = &hits[g_idx + 1];

        let same_ref = h_prev.entry_idx == h_next.entry_idx;
        let is_break = if same_ref || gap.contains(&b'\n') {
            true
        } else if let Some(dc) = dom_conn {
            gap != dc && gap.len() > dc.len() + 1
        } else {
            gap.len() > min_gap + 3
        };

        if is_break {
            clusters.push(cur);
            cur = vec![g_idx + 1];
        } else {
            cur.push(g_idx + 1);
        }
    }
    clusters.push(cur);

    let first_pos = hits[clusters[0][0]].pos;
    let last_end = hits[*clusters.last().unwrap().last().unwrap()].end;
    let lead_len = trim_edge_ws(&zone_bytes[0..first_pos]).len();
    let trail_len = trim_edge_ws(&zone_bytes[last_end..]).len();
    let suffix_mode = trail_len >= 3 && lead_len < 3;

    // Ne garde que les clusters qui référencent >= 2 causes distinctes avant le nettoyage d'affixes
    struct ValidCluster<'a> {
        raw_head: &'a [u8],
        clause_span: (usize, usize),
        mold_set: Vec<usize>,
        hop_depth: usize,
        ref_labels: Vec<Vec<u8>>,
    }

    let mut valid: Vec<ValidCluster<'_>> = Vec::new();
    for (ci, cl) in clusters.iter().enumerate() {
        let mut mold_set: Vec<usize> = Vec::new();
        let mut max_child_hop = 1usize;
        let mut ref_labels: Vec<Vec<u8>> = Vec::new();
        for &hi in cl {
            let entry = &entries[hits[hi].entry_idx];
            if entry.hop > max_child_hop {
                max_child_hop = entry.hop;
            }
            ref_labels.push(entry.key.clone());
            for &m in &entry.molds {
                if !mold_set.contains(&m) {
                    mold_set.push(m);
                }
            }
        }
        mold_set.sort_unstable();
        if mold_set.len() < 2 {
            continue;
        }

        let c_start = hits[cl[0]].pos;
        let c_end = hits[*cl.last().unwrap()].end;
        let prev_end = if ci == 0 {
            0
        } else {
            hits[*clusters[ci - 1].last().unwrap()].end
        };
        let next_start = if ci + 1 == clusters.len() {
            zone_bytes.len()
        } else {
            hits[clusters[ci + 1][0]].pos
        };

        let (raw_head, clause_span) = if suffix_mode {
            (
                &zone_bytes[c_end..next_start],
                (zone_offset + c_start, zone_offset + next_start),
            )
        } else {
            (
                &zone_bytes[prev_end..c_start],
                (zone_offset + prev_end, zone_offset + c_end),
            )
        };

        if trim_edge_ws(raw_head).len() >= 2 {
            valid.push(ValidCluster {
                raw_head,
                clause_span,
                mold_set,
                hop_depth: max_child_hop + 1,
                ref_labels,
            });
        }
    }

    if valid.is_empty() {
        return Vec::new();
    }
    let raw_heads: Vec<&[u8]> = valid.iter().map(|v| v.raw_head).collect();
    let clean_heads = strip_common_head_affixes(&raw_heads);

    let mut out = Vec::with_capacity(valid.len());
    for (vc, name_bytes) in valid.into_iter().zip(clean_heads.into_iter()) {
        if name_bytes.len() >= 2 {
            out.push(ComposedConcept {
                name_bytes,
                clause_span: vc.clause_span,
                mold_set: vc.mold_set,
                hop_depth: vc.hop_depth,
                ref_labels: vc.ref_labels,
            });
        }
    }
    out
}

pub fn analyze_multihop(graph: &PhiGraph, bytes: &[u8]) -> Option<MultiHopAnalysis> {
    let raw_regimes = discover_causal_regimes(graph, bytes, true);
    if raw_regimes.len() < 2 {
        return None;
    }

    let (regimes, demoted_meta_count) = demote_metalanguage_regimes(bytes, &raw_regimes);
    if regimes.len() < 2 {
        return None;
    }

    // 1. Regroupe les régimes finaux qui forment une cible multi-causes (couture <= 8 octets)
    let k = regimes.len();
    let mut split_t = k - 1;
    while split_t >= 3 && (k - split_t) < 4 {
        let cand_idx = split_t - 1;
        let gap_to_next = regimes[split_t]
            .core_start
            .saturating_sub(regimes[cand_idx].core_end);
        if gap_to_next > 8 {
            break;
        }
        let prev_end = regimes[cand_idx - 1].core_end;
        let gap_before = regimes[cand_idx].core_start.saturating_sub(prev_end);
        // Un composant de cible est soit après un bloc définitionnel (>= 20), soit collé (<= 8)
        if gap_before > 8 && gap_before < 20 {
            break;
        }
        let earlier = &regimes[0..cand_idx];
        let proj = project_span_to_molds(
            graph,
            bytes,
            (regimes[cand_idx].core_start, regimes[cand_idx].core_end),
            earlier,
        );
        let strong_match = proj
            .first()
            .is_some_and(|m| m.score >= 0.62 && m.cos_sim >= 0.40);
        if strong_match {
            split_t = cand_idx;
        } else {
            break;
        }
    }

    let mold_regimes = regimes[0..split_t].to_vec();
    let mut target_spans: Vec<(usize, usize)> = regimes[split_t..]
        .iter()
        .map(|r| (r.core_start, r.core_end))
        .collect();

    // Fusionne deux fragments cibles adjacents s'ils relèvent de la même cause (sur-découpe)
    if target_spans.len() >= 2 {
        let mut merged_t: Vec<(usize, usize)> = vec![target_spans[0]];
        for &(s, e) in &target_spans[1..] {
            let prev = *merged_t.last().unwrap();
            let h_prev = graph
                .abduce(&graph.encode_bytes(&bytes[prev.0..prev.1]), None)
                .final_state;
            let h_cur = graph
                .abduce(&graph.encode_bytes(&bytes[s..e]), None)
                .final_state;
            let ov = byte_overlap_ratio(&bytes[prev.0..prev.1], &bytes[s..e]);
            if cosine_sim(&h_prev, &h_cur) > 0.72 && ov > 0.50 {
                merged_t.last_mut().unwrap().1 = e;
            } else {
                merged_t.push((s, e));
            }
        }
        target_spans = merged_t;
    }

    // 2. Si la cible est encore monolithique, tente une décomposition guidée par le résidu
    let mut residual_info: Option<(f32, f32, f32)> = None;
    if target_spans.len() == 1 {
        let (ts, te) = target_spans[0];
        let dec = decompose_by_residual(graph, &bytes[ts..te]);
        if dec.parts.len() >= 2 {
            target_spans = dec
                .parts
                .into_iter()
                .map(|(a, b)| (ts + a, ts + b))
                .collect();
            residual_info = Some((dec.single_err, dec.split_err, dec.cross_cos));
        }
    } else if target_spans.len() >= 2 {
        let (s0, e0) = target_spans[0];
        let (s1, e1) = target_spans[1];
        let full_err = graph
            .abduce(&graph.encode_bytes(&bytes[s0..e1]), None)
            .final_mean_err;
        let t0 = graph.abduce(&graph.encode_bytes(&bytes[s0..e0]), None);
        let t1 = graph.abduce(&graph.encode_bytes(&bytes[s1..e1]), None);
        let l0 = (e0 - s0) as f32;
        let l1 = (e1 - s1) as f32;
        let split_err = (l0 * t0.final_mean_err + l1 * t1.final_mean_err) / (l0 + l1).max(1.0);
        let cos_lr = cosine_sim(&t0.final_state, &t1.final_state);
        residual_info = Some((full_err, split_err, cos_lr));
    }

    // 3. Projection Hop-1 de chaque composante cible vers les moules primitifs
    let mut hop1_per_part: Vec<Vec<ProjectionMatch>> = Vec::with_capacity(target_spans.len());
    for &span in &target_spans {
        hop1_per_part.push(project_span_to_molds(graph, bytes, span, &mold_regimes));
    }

    // 4. Collecte des étiquettes de moules et des zones définitionnelles passives
    let mut tag_spans: Vec<(usize, usize)> = Vec::with_capacity(mold_regimes.len());
    let mut def_zones: Vec<(usize, usize)> = Vec::new();

    for i in 0..mold_regimes.len() {
        let prev_end = if i == 0 {
            0
        } else {
            mold_regimes[i - 1].core_end
        };
        let cs = mold_regimes[i].core_start;
        if cs.saturating_sub(prev_end) > 28 {
            def_zones.push((prev_end, cs - 24));
            tag_spans.push((cs - 24, cs));
        } else {
            tag_spans.push((prev_end, cs));
        }
    }

    let after_molds = mold_regimes.last().unwrap().core_end;
    let before_target = target_spans[0].0;
    if before_target.saturating_sub(after_molds) >= 10 {
        def_zones.push((after_molds, before_target));
    }
    let after_target = target_spans.last().unwrap().1;
    if bytes.len().saturating_sub(after_target) >= 10 {
        def_zones.push((after_target, bytes.len()));
    }

    // 5. Extraction des clés discriminantes + résolution récursive Hop-2 / Hop-3
    let mut mold_keys: Vec<Option<Vec<u8>>> = Vec::with_capacity(mold_regimes.len());
    let mut entries: Vec<KeyEntry> = Vec::new();
    for i in 0..mold_regimes.len() {
        let key_opt = find_discriminative_key(bytes, &tag_spans, i, &def_zones);
        if let Some(ref k) = key_opt {
            entries.push(KeyEntry {
                key: k.clone(),
                molds: vec![i],
                hop: 1,
                defined_in: None,
            });
        }
        mold_keys.push(key_opt);
    }

    let mut all_concepts: Vec<ComposedConcept> = Vec::new();
    if entries.len() >= 2 {
        for &(ds, de) in &def_zones {
            let found = extract_definitions_in_zone(&bytes[ds..de], ds, &entries);
            all_concepts.extend(found);
        }

        if !all_concepts.is_empty() {
            let mut ext_entries = entries.clone();
            for c in &all_concepts {
                if c.name_bytes.len() >= 3
                    && ext_entries.iter().all(|e| e.key != c.name_bytes)
                {
                    ext_entries.push(KeyEntry {
                        key: c.name_bytes.clone(),
                        molds: c.mold_set.clone(),
                        hop: c.hop_depth,
                        defined_in: Some(c.clause_span),
                    });
                }
            }
            if ext_entries.len() > entries.len() {
                for &(ds, de) in &def_zones {
                    let found3 = extract_definitions_in_zone(&bytes[ds..de], ds, &ext_entries);
                    for c3 in found3 {
                        if c3.hop_depth >= 3
                            && all_concepts
                                .iter()
                                .all(|old| old.name_bytes != c3.name_bytes)
                        {
                            all_concepts.push(c3);
                        }
                    }
                }
            }
        }
    }

    // 6. Scoring compositionnel (Couverture x Précision)
    let m_count = mold_regimes.len();
    let mut res_mat: Vec<Vec<f32>> = vec![vec![0.0f32; m_count]; target_spans.len()];
    for (c_idx, matches) in hop1_per_part.iter().enumerate() {
        for m in matches {
            res_mat[c_idx][m.dst_idx] = m.score;
        }
    }

    let mut hop2_matches: Vec<MultiHopMatch> = Vec::with_capacity(all_concepts.len());
    for concept in all_concepts {
        let mut cov_sum = 0.0f32;
        for row in &res_mat {
            let best_m = concept
                .mold_set
                .iter()
                .map(|&mi| row[mi])
                .fold(0.0f32, f32::max);
            cov_sum += best_m;
        }
        let coverage = cov_sum / (res_mat.len().max(1) as f32);

        let mut prec_sum = 0.0f32;
        for &mi in &concept.mold_set {
            let best_c = res_mat.iter().map(|row| row[mi]).fold(0.0f32, f32::max);
            prec_sum += best_c;
        }
        let precision = prec_sum / (concept.mold_set.len().max(1) as f32);

        let score = (coverage * precision).sqrt();
        hop2_matches.push(MultiHopMatch {
            concept,
            score,
            coverage,
            precision,
        });
    }
    hop2_matches.sort_by(|a, b| b.score.total_cmp(&a.score));

    let definitions_found = hop2_matches.len();
    Some(MultiHopAnalysis {
        mold_regimes,
        mold_keys,
        target_spans,
        residual_info,
        hop1_per_part,
        hop2_matches,
        definitions_found,
        demoted_meta_count,
    })
}

pub fn analyze_monolithic_residual(graph: &PhiGraph, bytes: &[u8]) -> ResidualSplit {
    let _ = SEARCH_STEPS;
    decompose_by_residual(graph, bytes)
}
