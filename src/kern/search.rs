use crate::kern::{db::*, emb};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use tantivy::{collector::TopDocs, query::*, schema::Field, Term};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode { Hybrid, Bm25, Vec, Sem }

pub fn bm25(db: &Db, q: &str, n: usize) -> Result<Vec<(usize, f32)>> {
    let q: &str = &q.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect::<String>(); // no operators / hyphen phrases
    let sch = db.idx.schema();
    let (ft, fb, ftr): (Field, Field, Field) = (sch.get_field("title")?, sch.get_field("body")?, sch.get_field("tr")?);
    let searcher = db.idx.reader()?.searcher();
    let mut qp = QueryParser::for_index(&db.idx, vec![ft, fb]);
    qp.set_field_boost(ft, std::env::var("WIKI_TITLEBOOST").ok().and_then(|x| x.parse().ok()).unwrap_or(2.0));
    let (plain, _) = qp.parse_query_lenient(q);
    let run = |q: &dyn Query| -> Result<Vec<(usize, f32)>> {
        searcher.search(q, &TopDocs::with_limit(n).order_by_score())?.into_iter().map(|(s, a)| {
            let col = searcher.segment_reader(a.segment_ord).fast_fields().u64("pid")?;
            Ok((col.first(a.doc_id).unwrap_or(0) as usize, s))
        }).collect()
    };
    let hits = run(&plain)?;
    // fuzzy only as fallback: words unseen in the index (typos); more words if few hits
    let toks = |name: &str, q: &str| { let mut an = db.idx.tokenizers().get(name).unwrap(); let mut ts = an.token_stream(q); let mut v = vec![]; while ts.advance() { v.push(ts.token().text.clone()); } v };
    let df = |f, t: &str| searcher.doc_freq(&Term::from_field_text(f, t));
    let mut fz: Vec<Box<dyn Query>> = vec![];
    let d = |t: &str| if t.chars().count() <= 4 { 1 } else { 2 };
    for t in toks("raw", q) {
        let Some(st) = toks("de", &t).pop() else { continue }; // stopword
        let (a, b) = (df(ftr, &t)?, df(fb, &st)?);
        if (a + b) == 0 || (hits.len() < 10 && a + b < 5) { fz.push(Box::new(FuzzyTermQuery::new(Term::from_field_text(ftr, &t), d(&t), true))); }
    }
    for t in toks("de", q) {
        if df(fb, &t)? == 0 || (hits.len() < 10 && df(fb, &t)? < 5) { fz.push(Box::new(FuzzyTermQuery::new(Term::from_field_text(fb, &t), d(&t), true))); }
    }
    if fz.is_empty() { return Ok(hits); }
    let all = BooleanQuery::new(vec![(Occur::Should, plain), (Occur::Should, Box::new(BoostQuery::new(Box::new(BooleanQuery::union(fz)), 15.0)))]);
    run(&all)
}

fn ptext(db: &Db, pid: usize) -> Option<(Art, String, String)> {
    let (sec, t) = db.store.get(pid)?;
    Some((db.art_of(pid), sec, t))
}

pub fn vec(db: &Db, q: &str, n: usize) -> Result<Vec<(usize, f32)>> {
    let Some(e) = &db.emb else { return Ok(vec![]) };
    let qv = e.embed(q);
    let qb = emb::bits(&qv, &db.mean);
    let w = qb.len();
    let words: &[u64] = unsafe { std::slice::from_raw_parts(db.vecs.as_ptr() as *const u64, db.vecs.len() / 8) };
    const CH: usize = 1 << 16;
    let mut cand: Vec<(u32, u32)> = words.par_chunks(CH * w).enumerate().flat_map(|(ci, c)| {
        let mut v: Vec<(u32, u32)> = c.chunks_exact(w).enumerate().map(|(i, p)| (p.iter().zip(&qb).map(|(a, b)| (a ^ b).count_ones()).sum(), (ci * CH + i) as u32)).collect();
        if v.len() > 100 { v.select_nth_unstable(100); v.truncate(100); }
        v
    }).collect();
    if cand.len() > 100 { cand.select_nth_unstable(100); cand.truncate(100); }
    // rescore with float cosine
    let mut r: Vec<(usize, f32)> = cand.par_iter().filter_map(|&(_, pid)| {
        let (a, sec, t) = ptext(db, pid as usize)?;
        Some((pid as usize, emb::cos(&qv, &e.embed(&format!("{}. {sec}. {t}", a.title)))))
    }).collect();
    r.sort_by(|a, b| b.1.total_cmp(&a.1));
    r.truncate(n);
    Ok(r)
}

/// Semantic lead hits as (first-passage id, score); empty when the layer is missing or the helper is down.
fn sem(db: &Db, q: &str, n: usize) -> Vec<(usize, f32, f32)> {
    let Some(s) = &db.sem else { return vec![] };
    match s.search(q, n) {
        Ok(r) => { if std::env::var("WIKI_DBG").is_ok() { for x in r.iter().take(3) { eprintln!("sem {:.3} {}", x.1, db.art(x.0).title); } } r }.into_iter().filter_map(|(a, sc)| { let a = db.art(a); (a.n > 0).then_some((a.first, sc, sc)) }).collect(),
        Err(e) => { eprintln!("sem: {e}"); vec![] }
    }
}

/// RRF weight of the lead-semantic list (BM25 and potion lists weigh 1).
const SEMW: f32 = 2.0;
/// e5 cosine above which the top semantic hit overrides uncorroborated title guesses.
const SEMVETO: f32 = 0.86;

pub struct Hit { pub pid: usize, pub score: f32, pub art: Art, pub sec: String, pub text: String }

const QW: &[&str] = &["wer","wie","was","wann","wo","wohin","woher","welche","welcher","welches","welchen","wieso","warum","weshalb","wurde","wurden","wird","werden","ist","sind","war","waren","hat","haben","hatte","der","die","das","dem","den","des","ein","eine","einer","einen","einem","eines","von","vom","in","im","an","am","auf","zu","zum","zur","mit","bei","nach","und","oder","es","man","mir","mich","sich","gibt","kann","kennst","sag","sage","erzähl","erzähle","bitte","mal","eigentlich","denn","so","da","dort","liegt","liegen","lag","heißt","heisst","bedeutet","genau","ungefähr","etwa","ca","viele","viel","für","über","aus","um","als","wem","wen","wessen","hier","gerade","heute","noch","auch","nur","ich","du","er","sie","wir","ihr","gross","groß","lang","hoch","alt","weit","tief","schwer","gestorben","geboren","starb","erfunden","erfand","erfinden","entdeckt","entdeckte","gebaut","gegründet","gründete","entwickelt","begann","anfing","stattfand","passiert","geschah","is"];
const ATTR: &[&str] = &["hauptstadt","einwohner","einwohnerzahl","fläche","flaeche","länge","laenge","höhe","hoehe","währung","bevölkerung","geburtstag","geburtsdatum","todestag","sterbedatum","hauptstädte"];

fn words(q: &str) -> Vec<String> { q.split(|c: char| !c.is_alphanumeric() && c != '-').filter(|w| !w.is_empty()).map(String::from).collect() }
fn lc(s: &str) -> String { s.to_lowercase() }
fn cap(s: &str) -> String { s.chars().next().unwrap().to_uppercase().chain(s.chars().skip(1)).collect() }

/// Entity tokens: query words minus question/filler/attribute words.
fn entity(q: &str) -> Vec<String> {
    words(q).into_iter().filter(|w| { let l = lc(w); !QW.contains(&l.as_str()) && !ATTR.contains(&l.as_str()) }).collect()
}

/// Article ids for the query entity: exact title of a word span (longest first) merged with a whole-word title scan, ranked by popularity.
fn title_cands(db: &Db, q: &str) -> Vec<usize> {
    let ws = words(q);
    let isq = |w: &String| QW.contains(&lc(w).as_str());
    let isa = |w: &String| ATTR.contains(&lc(w).as_str());
    let pop = |a: &Art| (1.0 + a.n as f32).ln() * if a.title.contains('(') { 0.6 } else { 1.0 };
    let mut best: Vec<(f32, usize)> = vec![];
    let mut multi = false;
    let nent = entity(q).len().max(1);
    for len in 1..=ws.len().min(5) {
        for st in 0..=ws.len() - len {
            let sp = &ws[st..st + len];
            if isq(&sp[0]) || isq(&sp[len - 1]) || (len == 1 && isa(&sp[0])) { continue; }
            let t = sp.join(" ");
            let al = db.alias(&t);
            for a in [db.find(&t), db.find(&cap(&t)), al].into_iter().flatten() {
                { if !a.title.starts_with("Liste") && !best.iter().any(|x| x.1 == a.id) { let prep = st > 0 && ["in", "von", "aus", "bei", "nach"].contains(&lc(&ws[st - 1]).as_str());
                best.push((pop(&a) * (1.0 + 0.5 * len as f32) * (len as f32 / nent as f32).min(1.0).powi(2) * if prep { 0.6 } else { 1.0 }, a.id)); if len > 1 { multi = true; } } }
            }
        }
    }
    if multi { best.retain(|x| db.art(x.1).title.contains(' ') || db.art(x.1).title.contains('-')); }
    let ent: Vec<String> = entity(q).iter().flat_map(|w| w.split('-').map(String::from).collect::<Vec<_>>()).filter(|w| !w.is_empty()).collect();
    if ent.is_empty() { best.sort_by(|a, b| b.0.total_cmp(&a.0)); return best.into_iter().take(2).map(|x| x.1).collect(); }
    let key = ent.iter().max_by_key(|w| w.chars().count()).unwrap();
    let mut offs: Vec<usize> = vec![];
    for k in [cap(key), lc(key)] { offs.extend(memchr::memmem::find_iter(&db.titles, k.as_bytes())); }
    offs.sort_unstable(); offs.dedup();
    let na = db.narts();
    let mut last = usize::MAX;
    let tstart = |i: usize| u32::from_le_bytes(db.arts[i * 16 + 8..][..4].try_into().unwrap()) as usize;
    for o in offs {
        let (mut lo, mut hi) = (0, na);
        while hi - lo > 1 { let m = (lo + hi) / 2; if tstart(m) <= o { lo = m } else { hi = m } }
        if lo == last { continue; }
        last = lo;
        let a = db.art(lo);
        if a.title.starts_with("Liste") || best.iter().any(|x| x.1 == a.id) { continue; }
        let tw: Vec<String> = words(a.title.split(" (").next().unwrap_or("")).iter().flat_map(|w| w.split('-').map(lc).collect::<Vec<_>>()).collect();
        let mut pen = 1.0;
        let all = ent.iter().all(|e| { let e = lc(e); tw.iter().any(|t| *t == e || (t.starts_with(&e) && t.len() <= e.len() + 2 && { pen = 0.7; true })) });
        if !all { continue; }
        best.push((pop(&a) / (tw.len() as f32).powf(0.3) * pen * if nent > 1 { 1.5 } else { 1.0 }, a.id));
    }
    best.sort_by(|a, b| b.0.total_cmp(&a.0));
    if std::env::var("WIKI_DBG").is_ok() { for b in best.iter().take(6) { eprintln!("cand {:.2} {}", b.0, db.art(b.1).title); } }
    best.into_iter().take(2).map(|x| x.1).collect()
}

fn norm(t: &str) -> String { t.chars().filter(|c| c.is_alphanumeric()).take(200).collect() }

pub fn ask(db: &Db, q0: &str, k: usize, mode: Mode) -> Result<Vec<Hit>> {
    let n = 50;
    // question words only add noise to retrieval
    let q: String = { let w: Vec<String> = words(q0).into_iter().filter(|w| !QW.contains(&lc(w).as_str())).collect(); if w.is_empty() { q0.into() } else { w.join(" ") } };
    let q = q.as_str();
    let (mut semtop, mut semarts) = (0f32, Vec::<usize>::new());
    let ranked: Vec<(usize, f32)> = match mode {
        Mode::Bm25 => bm25(db, q, n)?,
        Mode::Vec => vec(db, q0, n)?,
        Mode::Sem => sem(db, q0, n).into_iter().map(|x| (x.0, x.1)).collect(),
        Mode::Hybrid => {
            let (b, v) = rayon::join(|| bm25(db, q, n), || vec(db, q0, n));
            let sm = sem(db, q0, n);
            semtop = sm.first().map_or(0.0, |x| x.2);
            semarts = sm.iter().take(10).map(|x| db.art_of(x.0).id).collect();
            let sm: Vec<(usize, f32)> = sm.into_iter().map(|x| (x.0, x.1)).collect();
            let mut m: HashMap<usize, f32> = HashMap::new();
            let sw: f32 = std::env::var("WIKI_SEMW").ok().and_then(|x| x.parse().ok()).unwrap_or(SEMW);
            for (l, w) in [(b?, 1.0), (v?, 1.0), (sm, sw)] { for (r, (p, _)) in l.iter().enumerate() { *m.entry(*p).or_default() += w / (60.0 + r as f32 + 1.0); } }
            let mut v: Vec<_> = m.into_iter().collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            v
        }
    };
    let mut cands = if mode == Mode::Vec || mode == Mode::Sem || std::env::var("WIKI_NOCANDS").is_ok() { vec![] } else { title_cands(db, q0) };
    // a confident semantic top hit vetoes title guesses that the semantic layer does not corroborate (e.g. "1945" for a who-question)
    if semtop >= SEMVETO { cands.retain(|c| semarts.contains(c)); }
    let wl = lc(q0).contains("liste");
    let top = ranked.first().map(|x| x.1).unwrap_or(0.03);
    let mut hits: Vec<Hit> = vec![];
    let mut seen = std::collections::HashSet::new();
    let mut push = |pid: usize, score: f32, hits: &mut Vec<Hit>| {
        let Some((art, sec, text)) = ptext(db, pid) else { return };
        if !seen.insert(norm(&text)) { return; }
        hits.push(Hit { pid, score, art, sec, text });
    };
    for (pid, score) in ranked.iter() {
        let a = db.art_of(*pid);
        let mut s = *score;
        if !wl && a.title.starts_with("Liste") { s *= 0.5; }
        if let Some(ci) = cands.iter().position(|c| *c == a.id) { s *= 2.0 - 0.3 * ci as f32; }
        push(*pid, s, &mut hits);
    }
    // lead passage of the entity article(s) wins
    for (ci, c) in cands.iter().enumerate().take(2) {
        let a = db.art(*c);
        if a.n == 0 { continue; }
        let s = top * (3.0 - ci as f32);
        if let Some(h) = hits.iter_mut().find(|h| h.pid == a.first) { h.score = h.score.max(s); }
        else { push(a.first, s, &mut hits); }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.pid.cmp(&b.pid)));
    let mut per: HashMap<usize, usize> = HashMap::new();
    let mut out = vec![];
    for h in hits {
        let c = per.entry(h.art.id).or_default();
        if *c >= 2 { continue; }
        *c += 1;
        out.push(h);
        if out.len() == k { break; }
    }
    Ok(out)
}
