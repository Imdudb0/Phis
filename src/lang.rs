//! Bootstrap minimal de définitions de langue (le "vocabulaire moyen").
//! L'utilisateur peut fournir un fichier `.lang` avec des lignes
//! `terme<TAB>définition` pour étendre.

pub fn bootstrap() -> Vec<(String, String)> {
    let defs: &[(&str, &str)] = &[
        ("ambiguity",      "a word or phrase that can be understood in more than one way"),
        ("sense",          "the meaning that a word or phrase has in context"),
        ("cause",          "something that produces an effect or result"),
        ("effect",         "a change that is produced by a cause"),
        ("word",           "a single unit of language that has meaning"),
        ("sentence",       "a group of words that expresses a complete thought"),
        ("meaning",        "what is meant by a word, a phrase, or a text"),
        ("language",       "a system of words used by a community to communicate"),
        ("text",           "written or printed words arranged in a coherent form"),
        ("explanation",    "a statement that makes something clear by describing its cause"),
        ("understanding",  "the ability to grasp the meaning of something"),
        ("search",         "an attempt to find something by looking carefully"),
        ("hypothesis",     "a proposed explanation made on the basis of limited evidence"),
        ("inference",      "a conclusion reached on the basis of evidence and reasoning"),
        ("abduction",      "a form of reasoning that seeks the most likely explanation"),
        ("interpretation", "the action of explaining the meaning of something"),
        ("context",        "the circumstances that form the setting for a word or text"),
        ("reference",      "the relation between a word and the thing it denotes"),
        ("relation",       "a connection between two or more things"),
        ("perception",     "the ability to see, hear, or become aware of something"),
        ("structure",      "the arrangement of and relations between the parts of something"),
        ("pattern",        "a repeated or regular way in which something happens"),
        ("comprehension",  "the action of understanding a subject fully"),
        ("cause-effect",   "the relation between an action and its result"),
    ];
    defs.iter().map(|(t, d)| (t.to_string(), d.to_string())).collect()
}

pub fn load_lang_file(path: &str) -> std::io::Result<Vec<(String, String)>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = vec![];
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some((t, d)) = line.split_once('\t') {
            out.push((t.trim().to_string(), d.trim().to_string()));
        }
    }
    Ok(out)
}