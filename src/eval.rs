//! Protocole scientifique v0.4.0 :
//! - Étalonnage PTGRS (C connue post-hoc)
//! - Échelle A : PTGRS OOD -> PTGRS Non-Séquentiel (clauses mélangées) -> Mini-Dict -> Mini-Langue
//! - Invariance bijective exacte ET stabilité au réordonnancement des paragraphes/clauses
//! - Critère B : Projection référentielle & abstraction méta-relationnelle sur dictionnaire riche
//!   (0 mot commun vérifié automatiquement) + témoin négatif non couvert.

use std::collections::HashSet;

use crate::abduction::{SearchConfig, build_residue_augmentation, run_abductive_search};
use crate::baselines::{
    BytePredictorBaseline, LikelihoodSearchBaseline, mdl_diagnostic_score, normalized_log_likelihood,
};
use crate::math::{Rng, pearson_correlation, spearman_rank_correlation};
use crate::phi_sharp::{EvalOptions, evaluate_phi_sharp};
use crate::ptgrs::{
    cause_recovery_score, sample_eval_case, sample_mini_dictionary, sample_nonseq_eval_case,
};
use crate::substrate::{HypothesisState, PerceivedGraph, PhisharpNet};

#[derive(Clone, Debug)]
pub struct CalibrationReport {
    pub passed: bool,
    pub corr_phi_vs_cause_spearman: f64,
    pub corr_ll_vs_cause_spearman: f64,
    pub corr_mdl_vs_cause_spearman: f64,
    pub corr_phi_vs_cause_pearson: f64,
    pub mean_phi_ptgrs: f64,
    pub mean_phi_random_bytes: f64,
    pub mean_phi_trivial_copy: f64,
}

pub fn run_calibration(net: &PhisharpNet, seed: u64) -> CalibrationReport {
    let mut rng = Rng::new(seed);
    let n_cases = 24;
    let mut cause_scores = Vec::with_capacity(n_cases);
    let mut phi_scores = Vec::with_capacity(n_cases);
    let mut phi_full_budget_scores = Vec::with_capacity(n_cases);
    let mut ll_scores = Vec::with_capacity(n_cases);
    let mut mdl_scores = Vec::with_capacity(n_cases);
    let mut copy_scores = Vec::with_capacity(n_cases);

    for i in 0..n_cases {
        let ood = i >= n_cases / 2;
        let case = sample_eval_case(&mut rng, ood);
        let pg = PerceivedGraph::from_bytes(&case.observed_e);

        let budget = match i % 4 {
            0 => 0,
            1 => 3,
            2 => 7,
            _ => 12,
        };
        let res = run_abductive_search(
            net,
            &case.observed_e,
            SearchConfig {
                budget_steps: budget,
                disable_refutation: false,
                max_explain_len: 128,
            },
        );
        let res_full = run_abductive_search(net, &case.observed_e, SearchConfig::default());
        phi_full_budget_scores.push(res_full.final_report.phi_sharp);

        let rec = cause_recovery_score(&case, &res.final_state.sym_trans, &pg.unique_bytes);
        let ll = normalized_log_likelihood(&res.final_state, &pg);
        let mdl = mdl_diagnostic_score(&res.final_state, &pg);

        cause_scores.push(rec);
        phi_scores.push(res.final_report.phi_sharp);
        ll_scores.push(ll);
        mdl_scores.push(mdl);

        let h_copy = HypothesisState::trivial_identity_from(&pg);
        let rep_copy = evaluate_phi_sharp(&h_copy, &pg, EvalOptions::default());
        copy_scores.push(rep_copy.phi_sharp);
    }

    let mut rand_scores = Vec::with_capacity(12);
    for _ in 0..12 {
        let len = 180 + rng.usize_below(120);
        let rand_bytes: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let res = run_abductive_search(net, &rand_bytes, SearchConfig::default());
        rand_scores.push(res.final_report.phi_sharp);
    }

    let corr_phi_s = spearman_rank_correlation(&phi_scores, &cause_scores);
    let corr_ll_s = spearman_rank_correlation(&ll_scores, &cause_scores);
    let corr_mdl_s = spearman_rank_correlation(&mdl_scores, &cause_scores);
    let corr_phi_p = pearson_correlation(&phi_scores, &cause_scores);

    let mean_phi_ptgrs = phi_full_budget_scores.iter().sum::<f64>() / (phi_full_budget_scores.len() as f64);
    let mean_phi_random = rand_scores.iter().sum::<f64>() / (rand_scores.len() as f64);
    let mean_phi_copy = copy_scores.iter().sum::<f64>() / (copy_scores.len() as f64);

    let passed = corr_phi_s >= corr_ll_s
        && corr_phi_s > corr_mdl_s
        && mean_phi_ptgrs > mean_phi_random * 2.5
        && mean_phi_ptgrs > mean_phi_copy * 2.5;

    CalibrationReport {
        passed,
        corr_phi_vs_cause_spearman: corr_phi_s,
        corr_ll_vs_cause_spearman: corr_ll_s,
        corr_mdl_vs_cause_spearman: corr_mdl_s,
        corr_phi_vs_cause_pearson: corr_phi_p,
        mean_phi_ptgrs,
        mean_phi_random_bytes: mean_phi_random,
        mean_phi_trivial_copy: mean_phi_copy,
    }
}

fn sample_mini_lang_cases(rng: &mut Rng, count: usize) -> Vec<Vec<u8>> {
    let nouns = ["kora", "velu", "tano", "mira", "soli", "neva"];
    let verbs = ["plix", "drok", "zami", "frel"];
    let mods = ["biv", "quo", "nar", "tus"];
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        let mut buf = Vec::new();
        for _ in 0..8 {
            let n1 = nouns[rng.usize_below(nouns.len())];
            let m1 = mods[rng.usize_below(mods.len())];
            let v1 = verbs[rng.usize_below(verbs.len())];
            let n2 = nouns[rng.usize_below(nouns.len())];
            let line = format!("{} {} {} {};\n", m1, n1, v1, n2);
            buf.extend_from_slice(line.as_bytes());
        }
        let qn = nouns[rng.usize_below(nouns.len())];
        let qv = verbs[rng.usize_below(verbs.len())];
        let qm = mods[rng.usize_below(mods.len())];
        buf.extend_from_slice(format!("? {} {} {}", qm, qn, qv).as_bytes());
        out.push(buf);
    }
    out
}

#[derive(Clone, Debug)]
pub struct ReferentialEvalItem {
    pub title: &'static str,
    pub perceived_eff: Vec<u8>,
    pub query: &'static str,
    pub target_def: &'static str,
    pub distractors: Vec<&'static str>,
    pub reordered_eff: Vec<u8>,
    pub meaning_broken_eff: Vec<u8>,
}

fn tokenize_words_lower(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

pub fn verify_zero_common_words(a: &str, b: &str) -> usize {
    let wa = tokenize_words_lower(a);
    let wb = tokenize_words_lower(b);
    wa.intersection(&wb).count()
}

const RICH_ENCYCLOPEDIA: &[&str] = &[
    "language, a system of conventional spoken, manual (signed), or written symbols by means of which human beings, as members of a social group and participants in its culture, express themselves. The functions of language include communication, the expression of identity, play, imaginative expression, and emotional release.",
    "human being, a culture-bearing primate classified in the genus Homo, especially the species H. sapiens. Human beings are anatomically similar and related to the great apes but are distinguished by a more highly developed brain and a resultant capacity for articulate speech and abstract reasoning.",
    "social group, any set of human beings who either are, recently have been, or anticipate being in some kind of interrelation. The term group, or social group, has been used to designate many kinds of aggregations of humans.",
    "culture, behaviour peculiar to Homo sapiens, together with material objects used as an integral part of this behaviour. Thus, culture includes language, ideas, beliefs, customs, codes, institutions, tools, techniques, works of art, rituals, and ceremonies.",
    "communication, the exchange of meanings between individuals through a common system of symbols.",
    "Meaning, in philosophy and linguistics, is the sense of a linguistic expression, sometimes understood in contrast to its referent. For example, the expressions morning star and evening star have different meanings, though their referent (Venus) is the same.",
    "intellectual history, branch of history that deals with the historical propagation and dissemination of ideas. Intellectual history is closely related to the history of ideas, a branch of history that treats ideas as objects of formal analysis.",
    "linguistics, the scientific study of language. The philologist is concerned primarily with the historical development of languages as it is manifest in written texts and in the context of the associated literature and culture.",
    "history, discipline that studies the chronological record of events, usually attempting, on the basis of a critical examination of source materials, to explain events.",
    "Venus, second planet from the Sun and sixth in the solar system in size and mass. No planet approaches closer to Earth than Venus; at its nearest it is the closest large body to Earth other than the Moon.",
    "Sun, star around which Earth and the other components of the solar system revolve. It is the dominant body of the system, constituting more than 99 percent of its entire mass.",
];

pub fn build_referential_eval_cases() -> Vec<ReferentialEvalItem> {
    struct RawCase {
        title: &'static str,
        entries: &'static [&'static str],
        query: &'static str,
        broken_query: &'static str,
        target_def: &'static str,
        distractors: &'static [&'static str],
    }

    let covered_ambig_dict: &[&str] = &[
        "contradiction: affirmation simultanee d une proposition et de sa negation logique.",
        "tautologie: formule vraie sous toute assignation possible des variables.",
        "regarde: action visuelle pouvant lier un agent a une cible ou a un instrument.",
        "avec: preposition indiquant soit un accompagnement soit un moyen instrumental.",
        "jumelles: objet optique servant d instrument ou couple de personnes.",
        "instrument: outil permettant plusieurs rattachements dans une construction verbale.",
        "rattachement: lien syntaxique produisant des lectures distinctes selon sa portee.",
        "ambiguite: pluralite de lectures distinctes possibles pour un meme enonce syntaxique.",
        "metaphore: transfert analogique d un domaine concret vers un concept abstrait.",
    ];

    let raws = [
        RawCase {
            title: "Projection referentielle riche (Who are social? -> H. sapiens)",
            entries: RICH_ENCYCLOPEDIA,
            query: "Who are social?",
            broken_query: "Which planet orbits nearest the star?",
            target_def: "culture-bearing primate classified in the genus Homo, especially the species H. sapiens",
            distractors: &[
                "second planet from the Sun and sixth in the solar system in size and mass",
                "discipline that studies the chronological record of events",
                "branch of history that deals with the historical propagation and dissemination of ideas",
                "star around which Earth and the other components of the solar system revolve",
            ],
        },
        RawCase {
            title: "Projection referentielle (morning star -> Venus)",
            entries: RICH_ENCYCLOPEDIA,
            query: "morning star",
            broken_query: "historical propagation of ideas",
            target_def: "second planet from the Sun and sixth in the solar system in size and mass",
            distractors: &[
                "culture-bearing primate classified in the genus Homo, especially the species H. sapiens",
                "discipline that studies the chronological record of events",
                "the exchange of meanings between individuals through a common system of symbols",
                "the scientific study of language",
            ],
        },
        RawCase {
            title: "Abstraction meta-relationnelle couverte (ambiguite syntaxique FR)",
            entries: covered_ambig_dict,
            query: "Pierre regarde Marie avec des jumelles.",
            broken_query: "Une formule logique reste toujours vraie.",
            target_def: "pluralite de lectures distinctes possibles pour un meme enonce syntaxique",
            distractors: &[
                "affirmation simultanee d une proposition et de sa negation logique",
                "formule vraie sous toute assignation possible des variables",
                "transfert analogique d un domaine concret vers un concept abstrait",
            ],
        },
    ];

    let mut cases = Vec::new();
    for r in raws {
        let overlap = verify_zero_common_words(r.query, r.target_def);
        assert_eq!(
            overlap, 0,
            "Violation protocole B : {} mots communs entre '{}' et '{}'",
            overlap, r.query, r.target_def
        );

        let build_eff = |entries: &[&str], q: &str| -> Vec<u8> {
            let mut s = entries.join("\n\n");
            s.push_str("\n\n");
            s.push_str(q);
            s.into_bytes()
        };

        let perceived_eff = build_eff(r.entries, r.query);

        // Réordonnancement complet des entrées + placement de la requête au MILIEU (teste le non-séquentiel !)
        let mut rev_entries: Vec<&str> = r.entries.to_vec();
        rev_entries.reverse();
        let mid = rev_entries.len() / 2;
        let mut reordered_vec: Vec<&str> = Vec::with_capacity(rev_entries.len() + 1);
        reordered_vec.extend_from_slice(&rev_entries[..mid]);
        reordered_vec.push(r.query);
        reordered_vec.extend_from_slice(&rev_entries[mid..]);
        let reordered_eff = reordered_vec.join("\n\n").into_bytes();

        let meaning_broken_eff = build_eff(r.entries, r.broken_query);

        cases.push(ReferentialEvalItem {
            title: r.title,
            perceived_eff,
            query: r.query,
            target_def: r.target_def,
            distractors: r.distractors.to_vec(),
            reordered_eff,
            meaning_broken_eff,
        });
    }
    cases
}

fn relational_alignment_score(output: &[u8], candidate: &str) -> f64 {
    let c_bytes = candidate.as_bytes();
    if output.is_empty() || c_bytes.is_empty() {
        return 0.0;
    }
    let mut big_c = HashSet::new();
    for w in c_bytes.windows(2) {
        big_c.insert((w[0], w[1]));
    }
    let mut big_o = HashSet::new();
    for w in output.windows(2) {
        big_o.insert((w[0], w[1]));
    }
    let inter2 = big_c.intersection(&big_o).count() as f64;
    let union2 = big_c.union(&big_o).count().max(1) as f64;

    let mut tri_c = HashSet::new();
    for w in c_bytes.windows(3) {
        tri_c.insert((w[0], w[1], w[2]));
    }
    let mut tri_o = HashSet::new();
    for w in output.windows(3) {
        tri_o.insert((w[0], w[1], w[2]));
    }
    let inter3 = tri_c.intersection(&tri_o).count() as f64;
    let union3 = tri_c.union(&tri_o).count().max(1) as f64;

    0.50 * (inter2 / union2) + 0.50 * (inter3 / union3)
}

fn byte_sequence_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut sa = HashSet::new();
    for w in a.windows(2) {
        sa.insert((w[0], w[1]));
    }
    let mut sb = HashSet::new();
    for w in b.windows(2) {
        sb.insert((w[0], w[1]));
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count().max(1) as f64;
    inter / union
}

#[derive(Clone, Debug)]
pub struct FullEvalSummary {
    pub calibration: CalibrationReport,
    pub invariance_exact: bool,
    pub a_reached: bool,
    pub b_reached: bool,
    pub mrr_phisharp: f64,
    pub mrr_byte_pred: f64,
    pub mrr_ll_search: f64,
}

pub fn run_bijective_invariance_test(net: &PhisharpNet, input: &[u8], seed: u64) -> (bool, f64) {
    let mut rng = Rng::new(seed);
    let mut perm: Vec<u8> = (0u8..=255u8).collect();
    rng.shuffle(&mut perm);
    let mut inv_perm = [0u8; 256];
    for (orig, &mapped) in perm.iter().enumerate() {
        inv_perm[mapped as usize] = orig as u8;
    }

    let permuted_input: Vec<u8> = input.iter().map(|&b| perm[b as usize]).collect();
    let cfg = SearchConfig::default();

    let res_orig = run_abductive_search(net, input, cfg);
    let res_perm = run_abductive_search(net, &permuted_input, cfg);

    let mut max_phi_diff = 0.0_f64;
    for (p1, p2) in res_orig.trajectory.iter().zip(res_perm.trajectory.iter()) {
        let d = (p1.report.phi_sharp - p2.report.phi_sharp).abs();
        if d > max_phi_diff {
            max_phi_diff = d;
        }
    }

    let unpermuted_out: Vec<u8> = res_perm
        .explanation_bytes
        .iter()
        .map(|&b| inv_perm[b as usize])
        .collect();
    let exact_bytes = unpermuted_out == res_orig.explanation_bytes;
    let passed = exact_bytes && max_phi_diff < 1e-9;
    (passed, max_phi_diff)
}

pub fn run_full_protocol(net: &PhisharpNet, seed: u64) -> FullEvalSummary {
    println!("=== PROTOCOLE PHISHARP v0.4.0 (HORS ENTRAINEMENT, NON-SEQUENTIEL + MOTIFS) ===");
    println!("Non-reciprocite du modele : {:.4}", net.non_reciprocity_index());

    let cal = run_calibration(net, seed ^ 0x101);
    println!("\n[1. Etalonnage PTGRS (C connue post-hoc)]");
    println!(
        "  Spearman(phi_sharp, cause)={:.3} | Spearman(LL, cause)={:.3} | Spearman(MDL, cause)={:.3}",
        cal.corr_phi_vs_cause_spearman, cal.corr_ll_vs_cause_spearman, cal.corr_mdl_vs_cause_spearman
    );
    println!(
        "  phi_sharp(PTGRS, b=12)={:.4} | phi_sharp(octets aleatoires, b=12)={:.4} | phi_sharp(H:=E)={:.4}",
        cal.mean_phi_ptgrs, cal.mean_phi_random_bytes, cal.mean_phi_trivial_copy
    );
    println!("  Verdict etalonnage : {}", if cal.passed { "VALIDE" } else { "ECHOUE (A illisible)" });

    println!("\n[2. Entrainement des Baselines a armes egales (PTGRS uniquement)]");
    let byte_pred = BytePredictorBaseline::train_on_ptgrs(140, seed ^ 0x202);
    let ll_search = LikelihoodSearchBaseline::train_equal_compute(35, 8, 4, 10, seed ^ 0x303);

    let mut rng = Rng::new(seed ^ 0x404);
    let mut ood_rec_phi = 0.0;
    let mut ood_rec_bp = 0.0;
    let mut ood_rec_ll = 0.0;
    let mut ood_proj_phi = 0.0;
    let mut ood_proj_bp = 0.0;
    let mut ood_proj_ll = 0.0;
    let mut sum_gen = 0.0;
    let mut sum_disc = 0.0;
    let mut sum_proj = 0.0;
    let mut sum_sens_min = 0.0;
    let mut sum_sens_prod = 0.0;

    let n_ood = 10;
    for _ in 0..n_ood {
        let case = sample_eval_case(&mut rng, true);
        let pg = PerceivedGraph::from_bytes(&case.observed_e);
        let res = run_abductive_search(net, &case.observed_e, SearchConfig::default());

        ood_rec_phi += cause_recovery_score(&case, &res.final_state.sym_trans, &pg.unique_bytes);
        ood_proj_phi += res.final_report.projection;

        let bp_mat = byte_pred.predict_transition_matrix(&pg);
        ood_rec_bp += cause_recovery_score(&case, &bp_mat, &pg.unique_bytes);
        ood_proj_bp += byte_pred.projection_score(&pg);

        let (ll_h, _, ll_p, _) = ll_search.run(&case.observed_e, 12);
        ood_rec_ll += cause_recovery_score(&case, &ll_h.sym_trans, &pg.unique_bytes);
        ood_proj_ll += ll_p;

        sum_gen += res.final_report.generativity;
        sum_disc += res.final_report.discrimination;
        sum_proj += res.final_report.projection;
        sum_sens_min += res.final_report.sens_min;
        sum_sens_prod += res.final_report.sens_prod;
    }
    let nf = n_ood as f64;
    ood_rec_phi /= nf;
    ood_rec_bp /= nf;
    ood_rec_ll /= nf;
    ood_proj_phi /= nf;
    ood_proj_bp /= nf;
    ood_proj_ll /= nf;

    // Évaluation sur entrées NON-SÉQUENTIELLES PTGRS (clauses mélangées)
    let mut nonseq_phi = 0.0;
    let mut nonseq_rec = 0.0;
    let mut nonseq_sens_min = 0.0;
    for _ in 0..8 {
        let nc = sample_nonseq_eval_case(&mut rng, true);
        let pg = PerceivedGraph::from_bytes(&nc.observed_e);
        let res = run_abductive_search(net, &nc.observed_e, SearchConfig::default());
        nonseq_phi += res.final_report.phi_sharp;
        nonseq_rec += cause_recovery_score(&nc, &res.final_state.sym_trans, &pg.unique_bytes);
        nonseq_sens_min += res.final_report.sens_min;
    }
    nonseq_phi /= 8.0;
    nonseq_rec /= 8.0;
    nonseq_sens_min /= 8.0;

    let mut dict_phi = 0.0;
    let mut dict_rec_phi = 0.0;
    for _ in 0..8 {
        let dc = sample_mini_dictionary(&mut rng, true);
        let pg = PerceivedGraph::from_bytes(&dc.observed_e);
        let res = run_abductive_search(net, &dc.observed_e, SearchConfig::default());
        dict_phi += res.final_report.phi_sharp;
        dict_rec_phi += cause_recovery_score(&dc, &res.final_state.sym_trans, &pg.unique_bytes);
    }
    dict_phi /= 8.0;
    dict_rec_phi /= 8.0;

    let ml_cases = sample_mini_lang_cases(&mut rng, 6);
    let mut ml_phi = 0.0;
    for mc in &ml_cases {
        ml_phi += run_abductive_search(net, mc, SearchConfig::default()).final_report.phi_sharp;
    }
    ml_phi /= ml_cases.len() as f64;

    println!("\n[3. Critere A : OOD -> Non-Sequentiel -> Mini-Dict -> Mini-Langue & Baselines]");
    println!(
        "  PTGRS OOD       : Gen={:.3} Disc={:.3} Proj={:.3} | Sens_min={:.3} Sens_prod={:.3}",
        sum_gen / nf, sum_disc / nf, sum_proj / nf, sum_sens_min / nf, sum_sens_prod / nf
    );
    println!(
        "  Recup Cause OOD : Phisharp={:.3} | BytePred={:.3} | LLSearch={:.3}",
        ood_rec_phi, ood_rec_bp, ood_rec_ll
    );
    println!(
        "  Projection OOD  : Phisharp={:.3} | BytePred={:.3} | LLSearch={:.3}",
        ood_proj_phi, ood_proj_bp, ood_proj_ll
    );
    println!(
        "  PTGRS Non-Seq   : phi_sharp={:.3}, cause_rec={:.3}, sens_min={:.3}",
        nonseq_phi, nonseq_rec, nonseq_sens_min
    );
    println!("  Mini-Dict PTGRS : phi_sharp={:.3}, cause_rec={:.3}", dict_phi, dict_rec_phi);
    println!("  Mini-Langue     : phi_sharp={:.3}", ml_phi);

    let probe_case = sample_eval_case(&mut rng, true);
    let res_full = run_abductive_search(net, &probe_case.observed_e, SearchConfig::default());
    let res_b0 = run_abductive_search(
        net,
        &probe_case.observed_e,
        SearchConfig { budget_steps: 0, disable_refutation: false, max_explain_len: 128 },
    );
    let res_noref = run_abductive_search(
        net,
        &probe_case.observed_e,
        SearchConfig { budget_steps: 12, disable_refutation: true, max_explain_len: 128 },
    );

    println!("\n  Courbe Anytime phi_sharp(t) sur OOD :");
    for pt in &res_full.trajectory {
        if pt.step % 3 == 0 || pt.step == res_full.trajectory.len() - 1 {
            println!(
                "    t={:2} : phi_sharp={:.4} (mip={:.4}, sens_min={:.3}, res={:.3})",
                pt.step, pt.report.phi_sharp, pt.report.phi_iit_mip, pt.report.sens_min, pt.report.residue_score
            );
        }
    }

    let aug_e = build_residue_augmentation(&probe_case.observed_e, &res_full.final_report);
    let res_aug = run_abductive_search(net, &aug_e, SearchConfig::default());
    let (inv_ok, inv_diff) = run_bijective_invariance_test(net, &probe_case.observed_e, seed ^ 0x777);

    println!("  Ablations :");
    println!(
        "    budget=12 : phi_sharp={:.4}, sens_min={:.4}, stab_ref={:.4}",
        res_full.final_report.phi_sharp, res_full.final_report.sens_min, res_full.final_report.stab_refutation
    );
    println!("    budget=0  : phi_sharp={:.4}, sens_min={:.4}", res_b0.final_report.phi_sharp, res_b0.final_report.sens_min);
    println!(
        "    no-refut  : phi_sharp={:.4}, stab_ref={:.4}",
        res_noref.final_report.phi_sharp, res_noref.final_report.stab_refutation
    );
    println!(
        "  Test Residu : phi_sharp(E)={:.4} -> phi_sharp(E + residu)={:.4}",
        res_full.final_report.phi_sharp, res_aug.final_report.phi_sharp
    );
    println!("  Invariance Bijective : exact_equivariant={} (max_delta_phi={:.2e})", inv_ok, inv_diff);

    let trajectory_grows = res_full.final_report.phi_sharp > res_b0.final_report.phi_sharp + 0.02;
    let refut_matters = res_full.final_report.phi_sharp > res_noref.final_report.phi_sharp + 0.002;
    let beats_baselines_a = ood_rec_phi >= ood_rec_ll && ood_proj_phi >= ood_proj_bp;
    let residue_ok = res_aug.final_report.phi_sharp >= res_full.final_report.phi_sharp;
    let a_reached = cal.passed && trajectory_grows && refut_matters && inv_ok && beats_baselines_a && residue_ok;

    println!("\n[4. Critere B : Projection referentielle & Abstraction meta-relationnelle (0 mot commun, NL jamais entrainee)]");
    let nl_cases = build_referential_eval_cases();
    let mut sum_mrr_phi = 0.0;
    let mut sum_mrr_bp = 0.0;
    let mut sum_mrr_ll = 0.0;
    let mut reorder_stab_sum = 0.0;
    let mut pert_shift_sum = 0.0;

    for (idx, item) in nl_cases.iter().enumerate() {
        let res_nl = run_abductive_search(net, &item.perceived_eff, SearchConfig::default());
        let res_reord = run_abductive_search(net, &item.reordered_eff, SearchConfig::default());
        let res_broken = run_abductive_search(net, &item.meaning_broken_eff, SearchConfig::default());

        let bp_out = byte_pred.emit_continuation(&item.perceived_eff, 160);
        let (_, _, _, ll_out) = ll_search.run(&item.perceived_eff, 12);

        let rank_output = |out: &[u8]| -> usize {
            let target_score = relational_alignment_score(out, item.target_def);
            let mut rank = 1usize;
            for d in &item.distractors {
                if relational_alignment_score(out, d) >= target_score {
                    rank += 1;
                }
            }
            rank
        };

        let r_phi = rank_output(&res_nl.explanation_bytes);
        let r_bp = rank_output(&bp_out);
        let r_ll = rank_output(&ll_out);

        sum_mrr_phi += 1.0 / (r_phi as f64);
        sum_mrr_bp += 1.0 / (r_bp as f64);
        sum_mrr_ll += 1.0 / (r_ll as f64);

        let sim_reord = byte_sequence_similarity(&res_nl.explanation_bytes, &res_reord.explanation_bytes);
        let sim_broken = byte_sequence_similarity(&res_nl.explanation_bytes, &res_broken.explanation_bytes);
        reorder_stab_sum += sim_reord;
        pert_shift_sum += 1.0 - sim_broken;

        let holes_str: Vec<String> = res_nl
            .final_report
            .residue_holes
            .iter()
            .take(3)
            .map(|b| format!("\"{}\"", String::from_utf8_lossy(b).replace('\n', "\\n")))
            .collect();
        let preview = String::from_utf8_lossy(&res_nl.explanation_bytes);
        println!(
            "  Cas {} ({}) | 0 mot commun verifie | Rang cible: Phisharp={} BytePred={} LL={} | Stab reord={:.3} vs Rupture sens={:.3}",
            idx + 1, item.title, r_phi, r_bp, r_ll, sim_reord, 1.0 - sim_broken
        );
        println!("    [Trous residu] [{}]", holes_str.join(", "));
        println!("    [Paquet aveugle sortie] \"{}\"", preview.replace('\n', "\\n"));
    }

    let nb = nl_cases.len() as f64;
    let mrr_phi = sum_mrr_phi / nb;
    let mrr_bp = sum_mrr_bp / nb;
    let mrr_ll = sum_mrr_ll / nb;
    let mean_reord = reorder_stab_sum / nb;
    let mean_shift = pert_shift_sum / nb;

    println!(
        "\n  MRR Projection B : Phisharp={:.3} | BytePred={:.3} | LLSearch={:.3}",
        mrr_phi, mrr_bp, mrr_ll
    );
    println!(
        "  Sensibilite B    : Stabilite reordonnancement={:.3} | Changement si requete changee={:.3}",
        mean_reord, mean_shift
    );

    let b_reached = mrr_phi > 0.55 && mrr_phi >= mrr_bp && mrr_phi >= mrr_ll && mean_shift > (1.0 - mean_reord);

    println!("\n[5. Lecture croisee]");
    println!("  A atteint : {} | B atteint : {}", a_reached, b_reached);
    let verdict = match (a_reached, b_reached) {
        (true, true) => "These soutenue",
        (true, false) => "Moitie abductive soutenue ; 'l'explication vient du sens' refutee dans sa forme actuelle",
        (false, true) => "Pattern matching ou heuristique : le succes s'explique sans recherche",
        (false, false) => "Sens non trouve ; seconde moitie non testee",
    };
    println!("  VERDICT : {}", verdict);

    FullEvalSummary {
        calibration: cal,
        invariance_exact: inv_ok,
        a_reached,
        b_reached,
        mrr_phisharp: mrr_phi,
        mrr_byte_pred: mrr_bp,
        mrr_ll_search: mrr_ll,
    }
}