use crate::model::Model;
use crate::percept::Percept;
use crate::phi::{Dir, Dist, Face, PHI_EPS};
use crate::resid::{Residual, MASS_EPS};
use crate::sharp::{Analysis, ComplexStruct, FrameResult, Unit};
use crate::util::{show_byte, show_bytes};
#[derive(Clone, Debug)]
pub struct RenderCfg {
    pub width: usize,
    pub curve_rows: usize,
    pub list_cap: usize,
    pub top_bytes: usize,
    pub frames_detail: usize,
    pub draws: Option<usize>,
}
impl Default for RenderCfg {
    fn default() -> RenderCfg {
        RenderCfg {
            width: 72,
            curve_rows: 24,
            list_cap: 0,
            top_bytes: 8,
            frames_detail: 0,
            draws: None,
        }
    }
}
pub fn unit_name(u: Unit) -> &'static str {
    match u {
        Unit::Bits => "bits",
        Unit::Symbols => "symbols",
    }
}
pub fn found_within_budget(an: &Analysis) -> bool {
    let mut any = false;
    for f in &an.frames {
        if f.phi > PHI_EPS {
            any = true;
            if !f.plateau {
                return false;
            }
        }
    }
    any
}
pub fn one_line(an: &Analysis) -> String {
    format!(
        "phi_sharp {:.4} {} (d {:.4} r {:.4})  A {:.3}  area {:.4}  t* {}  found {}  frames {}",
        an.phi,
        unit_name(an.unit),
        an.sum_d,
        an.sum_r,
        an.abstraction,
        an.area,
        an.t_star(),
        if found_within_budget(an) { "yes" } else { "no" },
        an.frames.len()
    )
}
fn pad(s: &str, w: usize) -> String {
    let n = s.chars().count();
    let mut r = String::from(s);
    for _ in n..w {
        r.push(' ');
    }
    r
}
const RAMP: [char; 6] = ['.', ':', '+', '*', '#', '@'];
pub fn glyph(mass: f64, max: f64) -> char {
    if !mass.is_finite() || !max.is_finite() || mass <= MASS_EPS || max <= MASS_EPS {
        return ' ';
    }
    let r = (mass / max).clamp(0.0, 1.0);
    let k = (r * (RAMP.len() - 1) as f64).round() as usize;
    RAMP[k.min(RAMP.len() - 1)]
}
fn runs(e: &Percept, pos: &[usize]) -> String {
    let mut out = String::new();
    let mut i = 0usize;
    while i < pos.len() {
        let mut j = i;
        while j + 1 < pos.len() && pos[j + 1] == pos[j] + 1 {
            j += 1;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        let mut bytes = String::new();
        for &p in &pos[i..=j] {
            match e.raw.get(p) {
                Some(&b) => bytes.push_str(&show_byte(b)),
                None => bytes.push('?'),
            }
        }
        if i == j {
            out.push_str(&format!("{}={}", pos[i], bytes));
        } else {
            out.push_str(&format!("{}..{}={}", pos[i], pos[j], bytes));
        }
        i = j + 1;
    }
    if out.is_empty() {
        out.push('-');
    }
    out
}
fn agreed(dists: &[Dist], members: &[usize], x: usize) -> Vec<u16> {
    let mut common: Option<Vec<u16>> = None;
    for &mi in members {
        let mut here: Vec<u16> = Vec::new();
        if let Some(d) = dists.get(mi) {
            for f in d.cause.iter().chain(d.effect.iter()) {
                if let Ok(i) = f.nodes.binary_search(&x) {
                    for s in &f.states {
                        if let Some(&v) = s.get(i) {
                            if !here.contains(&v) {
                                here.push(v);
                            }
                        }
                    }
                }
            }
        }
        common = Some(match common {
            None => here,
            Some(c) => c.into_iter().filter(|v| here.contains(v)).collect(),
        });
    }
    let mut v = common.unwrap_or_default();
    v.sort_unstable();
    v
}
fn pick_rows(n: usize, t_star: usize, rows: usize) -> Vec<usize> {
    if rows == 0 || n <= rows {
        return (0..n).collect();
    }
    let rows = rows.max(3);
    let mut v: Vec<usize> = (0..rows).map(|k| k * (n - 1) / (rows - 1)).collect();
    v.push(t_star.min(n - 1));
    v.sort_unstable();
    v.dedup();
    v
}
struct Out {
    s: String,
}
impl Out {
    fn new() -> Out {
        Out { s: String::new() }
    }
    fn ln(&mut self, line: &str) {
        self.s.push_str(line);
        self.s.push('\n');
    }
    fn blank(&mut self) {
        self.s.push('\n');
    }
}
pub fn strip(an: &Analysis, e: &Percept, width: usize) -> String {
    let n = e.len();
    if n == 0 {
        return "  (E is empty)\n".to_string();
    }
    let w = width.max(24);
    let mut read: Vec<Option<u16>> = vec![None; n];
    for fr in &an.frames {
        for k in 0..fr.len {
            let p = fr.start + k;
            if p < n {
                read[p] = fr.state.get(k).copied();
            }
        }
    }
    let mx = an.resid.by_pos.iter().cloned().fold(0.0f64, f64::max);
    let mut out = String::new();
    let mut p = 0usize;
    while p < n {
        let first = p;
        let (mut re, mut rs, mut rr) = (String::new(), String::new(), String::new());
        let mut used = 0usize;
        while p < n {
            let ce = show_byte(e.raw[p]);
            let cs = match read[p] {
                Some(v) if v == e.codes[p] => "\u{b7}".to_string(),
                Some(v) => match e.alphabet.get(v as usize) {
                    Some(&b) => show_byte(b),
                    None => "?".to_string(),
                },
                None => "?".to_string(),
            };
            let cw = ce.chars().count().max(cs.chars().count()).max(1) + 1;
            if used > 0 && used + cw > w {
                break;
            }
            let mass = an.resid.by_pos.get(p).copied().unwrap_or(0.0);
            re.push_str(&pad(&ce, cw));
            rs.push_str(&pad(&cs, cw));
            rr.push_str(&pad(&glyph(mass, mx).to_string(), cw));
            used += cw;
            p += 1;
        }
        let num = first.to_string();
        let rows = [(num.as_str(), "E", &re), ("", "s*", &rs), ("", "rho", &rr)];
        for (n1, label, row) in rows {
            let line = format!("{:>6} {:<4} {}", n1, label, row);
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }
    out
}
pub fn curve(fr: &FrameResult, scale: f64, rows: usize) -> String {
    const BAR: usize = 24;
    let mut s = String::from("  phi_sharp(t):\n");
    let mx = fr.steps.iter().map(|x| x.phi).fold(0.0f64, f64::max);
    let mut prev: Option<usize> = None;
    for t in pick_rows(fr.steps.len(), fr.t_star, rows) {
        if let Some(p) = prev {
            if t > p + 1 {
                s.push_str("      ...\n");
            }
        }
        prev = Some(t);
        let st = &fr.steps[t];
        let filled = if mx > PHI_EPS {
            (((st.phi / mx) * BAR as f64).round() as usize).min(BAR)
        } else {
            0
        };
        let bar: String = (0..BAR).map(|k| if k < filled { '#' } else { '.' }).collect();
        s.push_str(&format!(
            "    t={:>3} |{}| {:>9.4}  d {:.4} r {:.4}  tau {} cx {}{}\n",
            t,
            bar,
            st.phi * scale,
            st.sum_d * scale,
            st.sum_r * scale,
            st.tau,
            st.n_cx,
            if t == fr.t_star { "   <- t*" } else { "" }
        ));
    }
    s
}
struct View<'a> {
    e: &'a Percept,
    fr: &'a FrameResult,
    np_eff: usize,
    scale: f64,
    unit: &'static str,
    cfg: &'a RenderCfg,
    full: bool,
}
impl<'a> View<'a> {
    fn num(&self, bits: f64) -> String {
        format!("{:.4}", bits * self.scale)
    }
    fn cap(&self, n: usize) -> usize {
        if self.cfg.list_cap == 0 { n } else { n.min(self.cfg.list_cap) }
    }
    fn sym(&self, v: u16) -> String {
        match self.e.alphabet.get(v as usize) {
            Some(&b) => show_byte(b),
            None => "?".to_string(),
        }
    }
    fn label(&self, j: usize) -> String {
        if j < self.np_eff { format!("p{}", j) } else { format!("l{}", j) }
    }
    fn label_pos(&self, j: usize) -> String {
        if j < self.np_eff {
            let pos = self.fr.start + j;
            let b = match self.e.raw.get(pos) {
                Some(&b) => show_byte(b),
                None => "?".to_string(),
            };
            format!("p{}({}:{})", j, pos, b)
        } else {
            format!("l{}", j)
        }
    }
    fn plain(&self, nodes: &[usize]) -> String {
        let v: Vec<String> = nodes.iter().map(|&j| self.label(j)).collect();
        v.join(" ")
    }
    fn with_state(&self, nodes: &[usize], st: &[u16]) -> String {
        let mut s = String::new();
        for (i, &j) in nodes.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&self.label(j));
            if let Some(&v) = st.get(i) {
                s.push('=');
                s.push_str(&self.sym(v));
            }
        }
        s
    }
    fn face(&self, f: &Face) -> String {
        let mut s = String::new();
        if f.states.is_empty() {
            s.push_str(&self.plain(&f.nodes));
        } else {
            for (k, st) in f.states.iter().enumerate() {
                if k > 0 {
                    s.push_str(" | ");
                }
                s.push_str(&self.with_state(&f.nodes, st));
            }
        }
        s.push_str(&format!("  phi={}", self.num(f.phi)));
        s
    }
    fn foot_of(&self, c: &ComplexStruct, i: usize) -> String {
        match c.foot.get(i) {
            Some(f) => runs(self.e, f),
            None => "-".to_string(),
        }
    }
    fn frame(&self, o: &mut Out, budget: usize) {
        let fr = self.fr;
        let window = match self.e.raw.get(fr.start..fr.start + fr.len) {
            Some(b) => show_bytes(b),
            None => String::new(),
        };
        let verdict = if fr.phi <= PHI_EPS {
            "no structure"
        } else if fr.plateau {
            "plateau before B"
        } else {
            "maximum only at B: no plateau before B"
        };
        o.ln(&format!(
            "== frame {}: E[{},{}) = {}   phi_sharp {} {}   t* = {} of B = {}  ({}) ==",
            fr.frame,
            fr.start,
            fr.start + fr.len,
            window,
            self.num(fr.phi),
            self.unit,
            fr.t_star,
            budget,
            verdict
        ));
        o.ln(&format!("  grain tau* = {}, {} complex(es) at t*", fr.st.tau, fr.st.complexes.len()));
        o.s.push_str(&curve(fr, self.scale, self.cfg.curve_rows));
        if fr.st.complexes.is_empty() {
            o.ln("  no complex at t*: nothing irreducible in this frame");
        }
        for (ci, c) in fr.st.complexes.iter().enumerate() {
            self.complex(o, ci, c);
        }
    }
    fn complex(&self, o: &mut Out, ci: usize, c: &ComplexStruct) {
        o.ln(&format!(
            "  -- complex {}: {}   phi_s {} ({})",
            ci + 1,
            self.plain(&c.sys.nodes),
            self.num(c.sys.phi_s),
            if c.sys.exact {
                "every directed cut enumerated"
            } else {
                "sampled cuts + local search: an upper bound"
            }
        ));
        if !c.sys.cut_src.is_empty() && !c.sys.cut_dst.is_empty() {
            o.ln(&format!(
                "     weakest cut: {} -> {}",
                self.plain(&c.sys.cut_src),
                self.plain(&c.sys.cut_dst)
            ));
        }
        let st: Vec<u16> = c
            .sys
            .nodes
            .iter()
            .map(|&j| self.fr.state.get(j).copied().unwrap_or(0))
            .collect();
        o.ln(&format!("     state at t*: {}", self.with_state(&c.sys.nodes, &st)));
        o.ln(&format!(
            "     sums: phi_d {}  phi_r {}  total {} {}  (phi_r between disjoint footprints: {})",
            self.num(c.sum_d),
            self.num(c.sum_r),
            self.num(c.phi()),
            self.unit,
            self.num(c.sum_r_cross)
        ));
        self.dist_list(o, c);
        self.half_list(o, c);
        self.rel_list(o, c);
        self.resid_list(o, c);
    }
    fn dist_list(&self, o: &mut Out, c: &ComplexStruct) {
        let mut order: Vec<usize> =
            (0..c.dists.len()).filter(|&i| c.dists[i].is_distinction()).collect();
        order.sort_by(|&a, &b| c.dists[b].phi_d.total_cmp(&c.dists[a].phi_d).then(a.cmp(&b)));
        o.ln(&format!(
            "     Dist_E: {} distinction(s), sum phi_d {}",
            order.len(),
            self.num(c.sum_d)
        ));
        let shown = self.cap(order.len());
        for &i in order.iter().take(shown) {
            let d = &c.dists[i];
            o.ln(&format!(
                "       d{}  m {}  phi_d {}  (c {} e {})  foot {}",
                i,
                self.with_state(&d.mech, &d.mech_state),
                self.num(d.phi_d),
                self.num(d.phi_c),
                self.num(d.phi_e),
                self.foot_of(c, i)
            ));
            for f in &d.cause {
                o.ln(&format!("            cause  {}", self.face(f)));
            }
            for f in &d.effect {
                o.ln(&format!("            effect {}", self.face(f)));
            }
        }
        if shown < order.len() {
            o.ln(&format!(
                "       ... {} more distinction(s) not shown (list_cap = {})",
                order.len() - shown,
                self.cfg.list_cap
            ));
        }
    }
    fn half_list(&self, o: &mut Out, c: &ComplexStruct) {
        let half: Vec<usize> =
            (0..c.dists.len()).filter(|&i| !c.dists[i].is_distinction()).collect();
        if half.is_empty() {
            return;
        }
        o.ln(&format!(
            "     half-distinctions: {} (one irreducible side, phi_d = 0: they feed the residual, never the sum)",
            half.len()
        ));
        let shown = self.cap(half.len());
        for &i in half.iter().take(shown) {
            let d = &c.dists[i];
            let (side, val) = if d.phi_c > d.phi_e { ("cause", d.phi_c) } else { ("effect", d.phi_e) };
            o.ln(&format!(
                "       d{}  m {}  {} side only: {}  foot {}",
                i,
                self.with_state(&d.mech, &d.mech_state),
                side,
                self.num(val),
                self.foot_of(c, i)
            ));
        }
        if shown < half.len() {
            o.ln(&format!(
                "       ... {} more half-distinction(s) not shown (list_cap = {})",
                half.len() - shown,
                self.cfg.list_cap
            ));
        }
    }
    fn rel_list(&self, o: &mut Out, c: &ComplexStruct) {
        let mut order: Vec<usize> = (0..c.rels.len()).collect();
        order.sort_by(|&a, &b| c.rels[b].phi.total_cmp(&c.rels[a].phi).then(a.cmp(&b)));
        o.ln(&format!(
            "     Rel_E: {} relation(s), sum phi_r {}, of which between disjoint footprints {}",
            c.rels.len(),
            self.num(c.sum_r),
            self.num(c.sum_r_cross)
        ));
        let shown = self.cap(order.len());
        for &i in order.iter().take(shown) {
            let r = &c.rels[i];
            let mem: Vec<String> = r.members.iter().map(|&k| format!("d{}", k)).collect();
            let feet: Vec<String> =
                r.members.iter().map(|&k| format!("[{}]", self.foot_of(c, k))).collect();
            let ov: Vec<String> = r
                .overlap
                .iter()
                .map(|&x| {
                    let names: Vec<String> =
                        agreed(&c.dists, &r.members, x).iter().map(|&v| self.sym(v)).collect();
                    if names.is_empty() {
                        format!("{}=?", self.label(x))
                    } else {
                        format!("{}={}", self.label(x), names.join("|"))
                    }
                })
                .collect();
            o.ln(&format!(
                "       r{}  phi_r {}  {}  {}  overlap {}  foot {}",
                i,
                self.num(r.phi),
                if r.cross { "cross" } else { "local" },
                mem.join("+"),
                ov.join(" "),
                feet.join(" ")
            ));
        }
        if shown < order.len() {
            o.ln(&format!(
                "       ... {} more relation(s) not shown (list_cap = {})",
                order.len() - shown,
                self.cfg.list_cap
            ));
        }
    }
    fn resid_list(&self, o: &mut Out, c: &ComplexStruct) {
        if !self.full {
            return;
        }
        let (mut rc, mut re) = (0.0f64, 0.0f64);
        for r in &c.resid {
            match r.weak {
                Dir::Cause => rc += r.rho,
                Dir::Effect => re += r.rho,
            }
        }
        o.ln(&format!(
            "     residual: {} mechanism(s), rho_cause {} (clarté) + rho_effet {} (expressivité)",
            c.resid.len(),
            self.num(rc),
            self.num(re)
        ));
        let mut order: Vec<usize> = (0..c.resid.len()).collect();
        order.sort_by(|&a, &b| c.resid[b].rho.total_cmp(&c.resid[a].rho).then(a.cmp(&b)));
        let shown = self.cap(order.len());
        for &i in order.iter().take(shown) {
            self.resid_entry(o, c, &c.resid[i]);
        }
        if shown < order.len() {
            o.ln(&format!(
                "       ... {} more mechanism(s) not shown (list_cap = {})",
                order.len() - shown,
                self.cfg.list_cap
            ));
        }
    }
    fn resid_entry(&self, o: &mut Out, c: &ComplexStruct, r: &Residual) {
        let ms = match c.dists.iter().find(|d| d.mech == r.mech) {
            Some(d) => self.with_state(&r.mech, &d.mech_state),
            None => self.plain(&r.mech),
        };
        let kind = match r.weak {
            Dir::Cause => "rho_cause",
            Dir::Effect => "rho_effet",
        };
        let mut extra = String::new();
        if r.from_strong {
            extra.push_str("  [weak side has no irreducible purview: nodes of the strong side]");
        }
        if r.ties > 1 {
            extra.push_str(&format!("  [{} tied states]", r.ties));
        }
        o.ln(&format!(
            "       m {}  {} {}  needs {}{}",
            ms,
            kind,
            self.num(r.rho),
            r.need(),
            extra
        ));
        let parts: Vec<String> = r
            .nodes
            .iter()
            .map(|&(j, share)| format!("{} {}", self.label_pos(j), self.num(share)))
            .collect();
        o.ln(&format!("            on {}", parts.join(" | ")));
    }
}
fn is_full(an: &Analysis) -> bool {
    an.frames
        .iter()
        .all(|f| f.st.complexes.iter().all(|c| c.text.len() == c.dists.len()))
}
fn counts(fr: &FrameResult) -> (usize, usize) {
    let mut nd = 0usize;
    let mut nr = 0usize;
    for c in &fr.st.complexes {
        nd += c.dists.iter().filter(|d| d.is_distinction()).count();
        nr += c.rels.len();
    }
    (nd, nr)
}
fn plateau_word(fr: &FrameResult) -> &'static str {
    if fr.phi <= PHI_EPS {
        "n/a"
    } else if fr.plateau {
        "yes"
    } else {
        "no"
    }
}
pub fn render(an: &Analysis, m: &Model, e: &Percept, cfg: &RenderCfg) -> String {
    let mut o = Out::new();
    let full = is_full(an);
    let budget = m.hy.budget;
    let u = unit_name(an.unit);
    let nf = an.frames.len();
    let rm = &an.resid;
    o.ln("phisharp report");
    match cfg.draws {
        Some(r) => o.ln(&format!("  model        : {}  (this reading: R = {})", m.describe(), r)),
        None => o.ln(&format!("  model        : {}", m.describe())),
    }
    o.ln(&format!(
        "  E            : {} bytes, |A(E)| = {}, {} frame(s) of at most {} byte(s)",
        an.len, an.a, nf, m.np
    ));
    match an.unit {
        Unit::Bits => o.ln("  unit         : bits (log2)"),
        Unit::Symbols => {
            if an.a >= 2 {
                o.ln(&format!(
                    "  unit         : symbols of A(E), 1 symbol = log2 {} = {:.4} bits",
                    an.a,
                    (an.a as f64).log2()
                ));
            } else {
                o.ln("  unit         : symbols of A(E) (fewer than 2 symbols: every phi is 0)");
            }
        }
    }
    o.ln(&format!(
        "  phi_sharp(E) : {:.4} {}  =  sum phi_d {:.4} + sum phi_r {:.4}",
        an.phi, u, an.sum_d, an.sum_r
    ));
    if an.abstraction > 0.0 {
        o.ln(&format!(
            "  abstraction  : {:.4}  (share of phi_sharp carried by relations between disjoint footprints in E: justesse recoupée)",
            an.abstraction
        ));
    } else {
        o.ln("  abstraction  : 0.0000  (no relation between disjoint footprints in E)");
    }
    o.ln(&format!(
        "  area         : {:.4} {}  (mean of the running maximum of phi_sharp(t) over t = 0..={})",
        an.area, u, budget
    ));
    if an.phi_bits <= PHI_EPS {
        o.ln("  verdict      : no irreducible structure anchored in E: phi_sharp = 0 under this model");
    } else if found_within_budget(an) {
        o.ln(&format!(
            "  verdict      : findable within B = {}: every frame with a structure reaches its plateau before B (largest t* = {})",
            budget,
            an.t_star()
        ));
    } else {
        o.ln(&format!(
            "  verdict      : not findable within B = {}: a frame with a structure reaches its maximum only at B",
            budget
        ));
    }
    if full {
        let on_bytes: f64 = rm.by_pos.iter().sum();
        o.ln(&format!(
            "  residual     : rho_cause {:.4} (clarté) + rho_effet {:.4} (expressivité) = {:.4} {};  on bytes of E {:.4}, on latent nodes {:.4}",
            rm.cause, rm.effect, rm.total(), u, on_bytes, rm.latent
        ));
        o.ln("                 filling the weak side of a distinction d gains at most rho(d), its strong side unchanged");
    } else {
        o.ln("  residual     : not computed (Detail::Lean)");
    }
    o.blank();
    o.ln("E, read back at t* (s*), and residual (rho), byte by byte:");
    o.s.push_str(&strip(an, e, cfg.width));
    let mx = rm.by_pos.iter().cloned().fold(0.0f64, f64::max);
    if full && mx > MASS_EPS {
        o.ln(&format!(
            "  rho: blank = none, then . : + * # @ up to {:.4} {};   s*: \u{b7} = same byte as E",
            mx, u
        ));
        o.ln("  most residual bytes of E:");
        for (p, mass) in rm.top(cfg.top_bytes) {
            let b = match e.raw.get(p) {
                Some(&b) => show_byte(b),
                None => "?".to_string(),
            };
            o.ln(&format!("    E[{}] = {}   rho {:.4} {}", p, b, mass, u));
        }
    }
    if nf > 1 {
        o.blank();
        o.ln("frames (each analysed on its own, in the alphabet of the whole E):");
        o.ln(&format!(
            "  {:>5}  {:<14}  {:>9}  {:>3}  {:<7}  {:>9}  {:>6}  {:>5}",
            "frame", "bytes", "phi_sharp", "t*", "plateau", "complexes", "Dist_E", "Rel_E"
        ));
        for fr in &an.frames {
            let (nd, nr) = counts(fr);
            o.ln(&format!(
                "  {:>5}  {:<14}  {:>9.4}  {:>3}  {:<7}  {:>9}  {:>6}  {:>5}",
                fr.frame,
                format!("[{},{})", fr.start, fr.start + fr.len),
                fr.phi * an.scale,
                fr.t_star,
                plateau_word(fr),
                fr.st.complexes.len(),
                nd,
                nr
            ));
        }
    }
    let limit = if cfg.frames_detail == 0 { nf } else { cfg.frames_detail.min(nf) };
    for fr in an.frames.iter().take(limit) {
        let v = View { e, fr, np_eff: m.np.min(fr.len), scale: an.scale, unit: u, cfg, full };
        o.blank();
        v.frame(&mut o, budget);
    }
    if limit < nf {
        o.blank();
        o.ln(&format!(
            "  ... {} more frame(s) not detailed (frames_detail = {})",
            nf - limit,
            cfg.frames_detail
        ));
    }
    o.blank();
    o.ln("legend");
    o.ln("  pK  perception node K of the frame (carries E[start+K]);  lK  any other node (latent, valued in A(E))");
    o.ln("  symbols are bytes of A(E): \u{2423} space, \\xNN non-printable;  states are those of s_t*");
    o.ln("  m  mechanism with its state;  cause / effect  maximally irreducible purviews with their specified states (ties kept, separated by |)");
    o.ln("  foot  bytes of E touched by the purviews, position=byte (runs a..b=bytes);  cross  footprints pairwise disjoint: a relation of meaning, not of bytes");
    o.ln("  estimator, not exact: a minimum over a subset of cuts overestimates phi_s (marked 'upper bound'); mechanisms and purviews are limited by kmax, qmax (header)");
    o.s
}