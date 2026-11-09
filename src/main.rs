use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::{collections::HashMap, fs, io::Write, path::{Path, PathBuf}, process::Command};
use kern::{build, db, search};
mod kern;
mod web;
mod dense;
mod paths;
mod pdf;
mod think;
mod rank;

const BASE: &str = "https://lib-lib.org";

#[derive(Parser)]
#[command(about = "RAG over the libertarian library lib-lib.org (German PDFs) with page citations")]
struct Cli {
    /// Data dir: src/ (downloads), records.jsonl, works.json, idx/
    /// default: ~/AI-Model/buch if it exists, else the platform data dir (e.g. ~/.local/share/buch, %APPDATA%\buch)
    #[arg(long, global = true, env = "BUCH_DIR")]
    dir: Option<String>,
    /// optional Model2Vec dir (legacy BM25+m2v fallback; not needed with `buch dense`)
    #[arg(long, global = true, env = "WIKI_MODEL")]
    model: Option<String>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download books.json + PDFs (EPUB only if no PDF); sequential, 2 MB/s cap, skips existing
    Fetch { #[arg(long)] src: Option<String> },
    /// PDF -> pages -> cleaned page records -> hybrid index
    Build { #[arg(long)] src: Option<String> },
    /// Retrieve passages and answer with citations via an OpenAI-compatible endpoint
    Ask {
        query: String,
        /// passages sent to the answer model (after rerank)
        #[arg(short, default_value_t = 10)] k: usize,
        #[arg(long)] json: bool,
        /// print passages + citations only
        #[arg(long)] no_llm: bool,
        /// override the answer endpoint (implies --answer big)
        #[arg(long, env = "BUCH_ENDPOINT")] endpoint: Option<String>,
        #[arg(long, env = "BUCH_MODEL")] model_name: Option<String>,
        #[arg(long = "llm-model", env = "BUCH_LLM")] llm: Option<String>,
        /// answer model: big (default, config.json endpoint, e.g. Gemma) | lfm3 | heretic (test: Gemma-4 E4B via buch-llm)
        #[arg(long)] answer: Option<String>,
        /// rerank: none | d1 | lfm3 | lfm3list | big (config key rerank, default none)
        #[arg(long)] rerank: Option<String>,
        /// print the model's leaked reasoning to stderr (dimmed)
        #[arg(long)] show_thinking: bool,
        /// strip leaked reasoning: auto | on (buffer until marker) | off (stream live); config key strip_thinking
        #[arg(long, env = "BUCH_STRIP_THINKING")] strip_thinking: Option<String>,
        /// skip LLM query expansion
        #[arg(long)] no_expand: bool,
    },
    /// write every passage as JSON line {pid,title,page,text}
    Dump { out: PathBuf },
    /// embed all passages with multilingual-e5-small (candle, CPU; downloads ~470 MB from Hugging Face on first use) into <dir>/dense; BUCH_GPU_WRAP wraps the run in a queue command
    Dense { model_dir: Option<PathBuf>, /// output dir (default <dir>/dense)
        #[arg(long)] out: Option<PathBuf> },
    /// JSON lines on stdin ({"q":..,"k":3}) -> JSON lines {"hits":[{citation,text,..}],"ms"} (no LLM)
    Serve,
    /// Local web UI (127.0.0.1 only; optional port registry: BUCH_PORTS command)
    Web { #[arg(long, default_value_t = 8100)] port: u16 },
    /// TSV: query<TAB>expected work substring[<TAB>page|page]; prints hit@1/3/6/10, context hit, page hit, latency
    Eval { file: PathBuf, /// use LLM query expansion
        #[arg(long)] expand: bool,
        /// none | d1 | lfm3 | lfm3list | big
        #[arg(long, default_value = "none")] rerank: String,
        /// candidates before rerank
        #[arg(long, default_value_t = 30)] ncand: usize,
        /// passages sent to the answer model
        #[arg(long, default_value_t = 10)] ntop: usize,
        /// use the old pipeline (no title/entity suppression)
        #[arg(long)] legacy: bool,
        /// retrieval: m2v | bm25dense | dense | all (env BUCH_RETR)
        #[arg(long)] retr: Option<String> },
}

pub use paths::ex;

fn enc(p: &str) -> String {
    p.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn short_author(a: &str) -> String { a.split_whitespace().last().unwrap_or(a).to_string() }

/// local relative path below src/ for a catalog path; rejects traversal
fn rel(p: &str) -> Option<PathBuf> {
    let p = p.trim_start_matches('/');
    (!p.is_empty() && p != "#" && !p.split('/').any(|c| c.is_empty() || c == ".." || c == ".")).then(|| PathBuf::from(p))
}

fn books(src: &Path) -> Result<Vec<Value>> {
    Ok(serde_json::from_slice(&fs::read(src.join("books.json")).context("no books.json; run `buch fetch`")?)?)
}

fn curl(url: &str, out: &Path) -> Result<()> {
    if let Some(d) = out.parent() { fs::create_dir_all(d)?; }
    let tmp = out.with_extension("part");
    let st = Command::new("curl").args(["-fsSL", "--limit-rate", "2M", "--max-time", "900", "-o"]).arg(&tmp).arg(url).status()?;
    if !st.success() { let _ = fs::remove_file(&tmp); bail!("curl failed for {url}"); }
    fs::rename(tmp, out)?;
    Ok(())
}

fn fetch(src: &Path) -> Result<()> {
    fs::create_dir_all(src)?;
    curl(&format!("{BASE}/src/books.json"), &src.join("books.json"))?;
    let (mut got, mut skip, mut fail) = (0, 0, 0);
    for b in books(src)? {
        let d = &b["downloads"];
        let (pdf, epub) = (d["pdf"].as_str().unwrap_or(""), d["epub"].as_str().unwrap_or(""));
        let Some(p) = rel(pdf).or_else(|| rel(epub)) else { eprintln!("no file: {}", b["title"]); continue };
        let dst = src.join(&p);
        if dst.exists() { skip += 1; continue; }
        eprintln!("fetch {}", p.display());
        match curl(&format!("{BASE}/{}", enc(&p.to_string_lossy())), &dst) { Ok(_) => got += 1, Err(e) => { eprintln!("  {e}"); fail += 1 } }
    }
    eprintln!("fetched {got}, existing {skip}, failed {fail}");
    Ok(())
}

fn norm_line(l: &str) -> String { l.trim().chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).collect::<String>().to_lowercase() }

/// join lines of one page: fix hyphenation, drop-caps ("D\ner"), soft hyphens
fn join(lines: &[&str]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i].trim().replace('\u{ad}', "");
        i += 1;
        if l.is_empty() { continue; }
        let nxt = lines.get(i).map(|s| s.trim()).unwrap_or("");
        if l.chars().count() == 1 && l.chars().all(|c| c.is_uppercase()) && nxt.chars().next().is_some_and(|c| c.is_lowercase()) {
            out.push_str(&l); // drop cap glued to next line
            continue;
        }
        let ends_dash = l.ends_with('-') && l.chars().rev().nth(1).is_some_and(|c| c.is_alphabetic());
        if ends_dash && nxt.chars().next().is_some_and(|c| c.is_lowercase()) { out.push_str(&l[..l.len() - 1]); }
        else { out.push_str(&l); if !out.ends_with(|c: char| c.is_alphabetic() || c == '-') || !nxt.is_empty() { out.push(' '); } }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn license(pages: &[(u64, String)]) -> String {
    for (_, t) in pages.iter().take(6) {
        let t = t.replace('\n', " ");
        if let Some(i) = t.find("Creative Commons") {
            let s: String = t[i..].chars().take(120).collect::<String>().replace("- ", "-").replace("  ", " ");
            return if let Some(j) = s.find(".0") { s[..j + 2].to_string() } else { s.split(['.', ',']).next().unwrap_or("Creative Commons").trim().to_string() };
        }
    }
    "unbekannt".into()
}

fn build_idx(dir: &Path, src: &Path, model: &Path) -> Result<()> {
    let no_vec = !model.join("model.safetensors").exists();
    if no_vec { eprintln!("no Model2Vec model: building BM25 index only (run `buch dense` for semantic search)"); }
    fs::create_dir_all(dir)?;
    let mut recs = std::io::BufWriter::new(fs::File::create(dir.join("records.jsonl"))?);
    let mut works = serde_json::Map::new();
    let mut n = 0;
    for b in books(src)? {
        let (title_raw, author) = (b["title"].as_str().unwrap_or(""), b["author"].as_str().unwrap_or(""));
        let Some(p) = rel(b["downloads"]["pdf"].as_str().unwrap_or("")) else { eprintln!("skip (no PDF): {author}: {title_raw}"); continue };
        let f = src.join(&p);
        if !f.exists() { eprintln!("skip (not fetched): {}", p.display()); continue; }
        let pages = match pdf::pages(&f) { Ok(p) => p, Err(e) => { eprintln!("skip {}: {e}", p.display()); continue } };
        let title = format!("{}: {}", short_author(author), title_raw);
        let url = format!("{BASE}/{}", enc(&p.to_string_lossy()));
        // running headers/footers: first/last 2 non-empty lines whose digit-normalised form recurs on >=20% of pages
        let edge = |t: &str| -> (Vec<usize>, Vec<String>) { let ls: Vec<String> = t.lines().map(String::from).collect(); let idx: Vec<usize> = ls.iter().enumerate().filter(|(_, l)| !l.trim().is_empty()).map(|(i, _)| i).collect(); (idx, ls) };
        let mut cnt: HashMap<String, usize> = HashMap::new();
        for (_, t) in &pages {
            let (idx, ls) = edge(t);
            let e: Vec<usize> = idx.iter().take(2).chain(idx.iter().rev().take(2)).copied().collect();
            let mut seen: Vec<String> = e.iter().map(|&i| norm_line(&ls[i])).collect();
            seen.sort(); seen.dedup();
            for s in seen { *cnt.entry(s).or_default() += 1; }
        }
        let thr = (pages.len() / 5).max(3);
        let lic = license(&pages);
        let mut kept = 0;
        for (pg, t) in &pages {
            let (idx, ls) = edge(t);
            let last = idx.len();
            let body: Vec<&str> = idx.iter().enumerate().filter(|(r, i)| {
                let (r, i) = (*r, **i);
                let edge = r < 2 || r + 2 >= last;
                let nl = norm_line(&ls[i]);
                !(edge && (nl.chars().all(|c| c == '#') || (pages.len() >= 6 && cnt.get(&nl).is_some_and(|&c| c >= thr))))
            }).map(|(_, &i)| ls[i].as_str()).collect();
            let txt = join(&body);
            if txt.split_whitespace().count() < 15 { continue; }
            let txt = txt.trim_start_matches('#');
            writeln!(recs, "{}", json!({"title": title, "url": format!("{url}#page={pg}"), "text": format!("# {title}\n# S. {pg}\n{txt}")}))?;
            kept += 1; n += 1;
        }
        eprintln!("{title}: {kept}/{} pages, license {lic}", pages.len());
        works.insert(title.clone(), json!({"author": author, "work": title_raw, "year": b["releaseDate"], "translator": b["translator"], "license": lic, "pdf": url, "pages": pages.len(),
            "page_note": "page = PDF page; printed page may differ by 1 in some works"}));
    }
    recs.flush()?;
    drop(recs);
    fs::write(dir.join("works.json"), serde_json::to_vec_pretty(&Value::Object(works))?)?;
    eprintln!("{n} page records; indexing...");
    build::run(&[dir.join("records.jsonl")], &dir.join("idx"), model, no_vec)
}

#[derive(Clone)]
pub struct Cite { pub citation: String, pub title: String, pub page: String, pub url: String, pub license: String, pub score: f32, pub text: String, pub author: String, pub year: Value, pub translator: Value }

fn mkcite(works: &Value, t: String, sec: &str, text: String, score: f32) -> Cite {
    let pg = sec.trim_start_matches("S. ").to_string();
    let w = &works[&t];
    let url = format!("{}#page={pg}", w["pdf"].as_str().unwrap_or(""));
    Cite { citation: format!("[{t}, S. {pg}]"), title: t, page: pg, url, license: w["license"].as_str().unwrap_or("unbekannt").into(), score, text, author: w["author"].as_str().unwrap_or("").into(), year: w["year"].clone(), translator: w["translator"].clone() }
}

fn hits(works: &Value, h: Vec<search::Hit>) -> Vec<Cite> { h.into_iter().map(|h| mkcite(works, h.art.title.clone(), &h.sec, h.text, h.score)).collect() }

static DENSE: std::sync::OnceLock<Option<dense::Dense>> = std::sync::OnceLock::new();

/// dense passage list (max 2 per work like search::ask); empty when no vectors / embedder down
fn dense_cites(db: &db::Db, works: &Value, q: &str, n: usize) -> Vec<Cite> {
    let Some(d) = dense_of(db) else { return vec![] };
    let r = match d.search(q, 200) { Ok(r) => r, Err(e) => { eprintln!("dense: {e}"); return vec![] } };
    let mut per: HashMap<usize, usize> = HashMap::new();
    let mut out = vec![];
    for (pid, s) in r {
        let Some((sec, t)) = db.store.get(pid) else { continue };
        let a = db.art_of(pid);
        let c = per.entry(a.id).or_default();
        if *c >= 2 { continue; }
        *c += 1;
        out.push(mkcite(works, a.title, &sec, t, s));
        if out.len() == n { break; }
    }
    out
}

/// BUCH_RETR (default bm25dense when vectors exist): m2v (BM25+Model2Vec, old) | bm25dense (BM25 + dense) | dense | all (m2v fused + dense)
pub fn retr_mode(db: &db::Db) -> String { std::env::var("BUCH_RETR").unwrap_or_else(|_| if dense_of(db).is_some() { "bm25dense" } else { "m2v" }.into()) }

fn dense_of(db: &db::Db) -> Option<&'static dense::Dense> {
    DENSE.get_or_init(|| dense::Dense::open(&std::env::var("BUCH_DENSE").map(PathBuf::from).unwrap_or(db.dir.parent().unwrap_or(&db.dir).join("dense")))).as_ref()
}

pub fn cites(db: &db::Db, works: &Value, q: &str, k: usize) -> Result<Vec<Cite>> {
    let mut m = retr_mode(db);
    if m != "m2v" && dense_of(db).is_none() { m = "m2v".into(); }
    let n = k.max(50);
    let wd: f32 = std::env::var("BUCH_WDENSE").ok().and_then(|x| x.parse().ok()).unwrap_or(1.0);
    let r = match m.as_str() {
        "bm25dense" => rrf(vec![(1.0, hits(works, search::ask(db, q, n, search::Mode::Bm25)?)), (wd, dense_cites(db, works, q, n))], k),
        "dense" => { let mut d = dense_cites(db, works, q, k); d.truncate(k); d }
        "all" => rrf(vec![(1.0, hits(works, search::ask(db, q, n, search::Mode::Hybrid)?)), (wd, dense_cites(db, works, q, n))], k),
        _ => hits(works, search::ask(db, q, k, search::Mode::Hybrid)?),
    };
    Ok(r)
}

pub fn cj(c: &Cite) -> Value { json!({"citation": c.citation, "title": c.title, "page": c.page, "url": c.url, "license": c.license, "score": c.score, "text": c.text, "author": c.author, "year": c.year, "translator": c.translator, "work": c.title.split_once(": ").map(|x| x.1).unwrap_or("")}) }

pub fn open(dir: &Path, model: &Path) -> Result<(db::Db, Value)> {
    unsafe {
        std::env::set_var("WIKI_NOSEM", "1");
        // work titles carry author names: no entity-title boosting for a book library
        if std::env::var("WIKI_TITLEBOOST").is_err() { std::env::set_var("WIKI_TITLEBOOST", "0.3"); }
        std::env::set_var("WIKI_NOCANDS", "1");
    }
    let works = serde_json::from_slice(&fs::read(dir.join("works.json")).context("no index; run `buch build`")?)?;
    Ok((db::Db::open(&dir.join("idx"), model)?, works))
}

pub const SYS: &str = "Du beantwortest Fragen anhand der Textstellen aus der libertären Bibliothek, die unter der Frage stehen. Nutze nur diese Textstellen und fasse zusammen, was die Autoren dazu sagen, auch über mehrere Stellen hinweg. Deckt der Text die Frage nur teilweise ab, gib wieder, was er sagt, und nenne, was fehlt. Nur wenn keine Stelle zum Thema passt, antworte: \"Das steht nicht in der Bibliothek.\" Gib die Position des Autors sachlich und neutral wieder (\"Hoppe argumentiert, …\"), ohne Zustimmung, Wertung oder eigene Meinung. Zitiere wichtige Sätze wörtlich in „…“ und setze danach die Quelle exakt so, wie sie bei der Passage steht, z. B. [Hoppe: Libertarismus richtig verstehen, S. 16]. Erfinde nichts. Antworte auf Deutsch in 3 bis 8 Sätzen.";

pub fn agent(conn: u64, total: u64) -> ureq::Agent {
    ureq::Agent::config_builder().timeout_connect(Some(std::time::Duration::from_secs(conn))).timeout_global(Some(std::time::Duration::from_secs(total))).http_status_as_error(false).build().into()
}

/// streaming chat completion; `f` gets answer tokens (reasoning split off); returns thinking
pub fn stream_chat(ag: &ureq::Agent, endpoint: &str, key: Option<&str>, model: &str, body: Value, mode: think::Mode, mut f: impl FnMut(&str)) -> Result<String> {
    use std::io::BufRead;
    let mut body = body;
    body["model"] = json!(model);
    body["stream"] = json!(true);
    let mut req = ag.post(&format!("{}/chat/completions", endpoint.trim_end_matches('/')));
    if let Some(k) = key { req = req.header("Authorization", format!("Bearer {k}")); }
    let resp = req.send_json(&body).context("LLM endpoint")?;
    if !resp.status().is_success() { bail!("LLM endpoint: HTTP {}", resp.status()); }
    let mut sp = think::Splitter::new(mode, 200, 200);
    for l in std::io::BufReader::new(resp.into_body().into_reader()).lines() {
        let l = l?;
        let Some(d) = l.strip_prefix("data:") else { continue };
        let d = d.trim();
        if d == "[DONE]" { break; }
        let Ok(v) = serde_json::from_str::<Value>(d) else { continue };
        let dl = &v["choices"][0]["delta"];
        if let Some(t) = dl["reasoning_content"].as_str().or(dl["reasoning"].as_str()) { sp.reasoning(t); }
        if let Some(t) = dl["content"].as_str() {
            let o = sp.push(t);
            if !o.is_empty() { f(&o); }
        }
    }
    let o = sp.finish();
    if !o.is_empty() { f(&o); }
    Ok(sp.thinking)
}

const EXP_SYS: &str = "Du hilfst bei der Volltextsuche in deutschen libertären Büchern (Rothbard, Hoppe, Mises, Spooner, Kinsella u. a.). Schreibe auf, mit welchen Wörtern ein Autor den gesuchten Inhalt im Buch wörtlich ausdrücken würde: 8 bis 12 konkrete deutsche Substantive, Verben und kurze Wendungen aus der Alltags- und Autorensprache. Keine Abkürzungen, keine abstrakten Fachbegriffe, keine Sätze. Nur das Format als Beispiel (anderes Thema): Frage: Wie backe ich Brot? Antwort: Teig Mehl Hefe Ofen kneten Kruste gehen lassen Backzeit. Antworte nur mit den Begriffen zur gestellten Frage, in einer Zeile.";

/// LLM query expansion; None on any failure (3 s connect / 15 s total)
pub fn expand(ep: &rank::Ep, q: &str) -> Option<String> {
    let body = json!({"temperature": 0, "max_tokens": 80, "messages": [{"role": "system", "content": EXP_SYS}, {"role": "user", "content": format!("Frage: {q}")}]});
    let mut out = String::new();
    stream_chat(&agent(3, 15), &ep.url, ep.key.as_deref(), &ep.model, body, think::Mode::resolve(std::env::var("BUCH_STRIP_THINKING").ok().as_deref(), &web::cfg()), |t| out.push_str(t)).ok()?;
    clean_terms(&out)
}

pub fn clean_terms(s: &str) -> Option<String> {
    let l = s.lines().map(str::trim).find(|l| !l.is_empty())?;
    let t: String = l.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { ' ' }).collect::<String>().split_whitespace().take_while(|w| !["Antwort", "Frage", "Begriffe", "Begriff", "Aber", "think", "Keine"].contains(w)).filter(|w| !w.chars().all(|c| c.is_ascii_digit())).take(12).collect::<Vec<_>>().join(" ");
    (!t.is_empty()).then_some(t)
}

/// weighted RRF (k=60) over ranked lists, dedupe by citation; score = fused score
pub fn rrf(lists: Vec<(f32, Vec<Cite>)>, k: usize) -> Vec<Cite> {
    let mut m: HashMap<String, (f32, Cite)> = HashMap::new();
    for (w, l) in lists { for (r, c) in l.into_iter().enumerate() {
        let s = w / (60.0 + r as f32 + 1.0);
        m.entry(c.citation.clone()).and_modify(|e| e.0 += s).or_insert((s, c));
    } }
    let mut v: Vec<_> = m.into_values().collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.citation.cmp(&b.1.citation)));
    v.into_iter().take(k).map(|(s, mut c)| { c.score = s; c }).collect()
}

/// retrieve with original query and (optional) expansion terms, fused
pub fn retrieve(db: &db::Db, works: &Value, q: &str, terms: Option<&str>, k: usize) -> Result<Vec<Cite>> {
    let Some(t) = terms else { return cites(db, works, q, k) };
    let n = k.max(50);
    Ok(rrf(vec![(1.0, cites(db, works, q, n)?), (1.0, cites(db, works, &format!("{q} {t}"), n)?)], k))
}

pub struct Found { pub terms: Option<String>, pub cands: Vec<Cite>, pub top: Vec<Cite>, pub ms: [u128; 3], pub warn: Vec<String> }

/// expand (fast model) -> retrieve `ncand` fused -> rerank to `ntop`. `ev(name, value)` reports progress; every model failure degrades gracefully
pub fn find(db: &db::Db, works: &Value, c: &Value, q: &str, expand_on: bool, kind: &str, ncand: usize, ntop: usize, ev: &mut dyn FnMut(&str, Value)) -> Result<Found> {
    let t = std::time::Instant::now();
    let mut warn = vec![];
    let mut terms = None;
    if expand_on {
        ev("status", json!("Suchbegriffe …"));
        match rank::fast() { Ok(ep) => terms = expand(&ep, q), Err(e) => warn.push(e.to_string()) }
        if let Some(t) = &terms { ev("terms", json!(t)); }
    }
    let t1 = t.elapsed().as_millis();
    let cands = retrieve(db, works, q, terms.as_deref(), ncand)?;
    let t2 = t.elapsed().as_millis();
    let mut top: Vec<Cite> = cands.iter().take(ntop).cloned().collect();
    if kind != "none" && cands.len() > 1 {
        ev("status", json!("Sortiere Treffer …"));
        match rank::rerank(c, kind, q, &cands, ntop) { Ok(r) => top = r, Err(e) => warn.push(format!("Sortierung ({kind}) fehlgeschlagen: {e}")) }
    }
    let t3 = t.elapsed().as_millis();
    Ok(Found { terms, cands, top, ms: [t1, t2 - t1, t3 - t2], warn })
}

/// chat body for the answer model: passages are cut to 2000 chars each
pub fn answer_body(model: &str, q: &str, cs: &[Cite], stream: bool) -> Value {
    let ctx: String = cs.iter().map(|c| { let t: String = c.text.chars().take(2000).collect(); format!("{}\n{}\n", c.citation, t) }).collect::<Vec<_>>().join("\n");
    json!({"model": model, "temperature": 0.2, "max_tokens": if model == rank::heretic_model() { 2000 } else { 900 }, "stream": stream, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": format!("Passagen:\n\n{ctx}\nFrage: {q}")}]})
}

fn main() -> Result<()> {
    let c = Cli::parse();
    let dir = c.dir.as_deref().map(ex).unwrap_or_else(paths::data_dir);
    let model = c.model.as_deref().map(ex).unwrap_or_else(|| ex("~/AI-Model/models/minishlab/potion-multilingual-128M"));
    match c.cmd {
        Cmd::Fetch { src } => fetch(&src.map(|s| ex(&s)).unwrap_or(dir.join("src")))?,
        Cmd::Build { src } => build_idx(&dir, &src.map(|s| ex(&s)).unwrap_or(dir.join("src")), &model)?,
        Cmd::Ask { query, k, json: js, no_llm, endpoint, model_name, llm: lm, answer, rerank, show_thinking, strip_thinking, no_expand } => {
            let (db, works) = open(&dir, &model)?;
            let c = web::cfg();
            let kind = rerank.unwrap_or_else(|| rank::rerank_kind(&c));
            let f = find(&db, &works, &c, &query, !(no_llm || no_expand) && rank::expand_on(&c), &kind, 30, k, &mut |n, v| if n == "terms" { eprintln!("\x1b[2mGesucht nach: {}\x1b[0m", v.as_str().unwrap_or("")) })?;
            for w in &f.warn { eprintln!("warn: {w}"); }
            eprintln!("\x1b[2m[{} ms Suchbegriffe, {} ms Suche, {} ms Sortierung ({kind})]\x1b[0m", f.ms[0], f.ms[1], f.ms[2]);
            let cs = &f.top;
            let srcs = || { let mut s: Vec<&str> = cs.iter().map(|c| c.citation.as_str()).collect(); s.sort(); s.dedup(); s.join("\n  ") };
            if no_llm {
                if js { println!("{}", json!({"answer": null, "terms": f.terms, "passages": cs.iter().map(cj).collect::<Vec<_>>()})); }
                else { for (i, c) in cs.iter().enumerate() { println!("[{}] {}  ({:.4})  Lizenz: {}\n    {}\n{}\n", i + 1, c.citation, c.score, c.license, c.url, c.text); } }
            } else {
                let akind = answer.clone().unwrap_or_else(|| rank::pick("BUCH_ANSWER", "answer", &c, "big"));
                let ep = if endpoint.is_some() || lm.is_some() || model_name.is_some() {
                    let mut e = rank::big(&c);
                    if let Some(u) = endpoint { e.url = u; }
                    if let Some(m) = lm.or(model_name) { e.model = m; }
                    e
                } else { rank::answer_ep_for(&akind, &c, || eprintln!("\x1b[2mLade Modell …\x1b[0m"))? };
                let heretic = akind == "heretic" && ep.model == rank::heretic_model();
                let body = answer_body(&ep.model, &query, cs, true);
                let mut ans = String::new();
                let mut so = std::io::stdout();
                let t0 = std::time::Instant::now();
                let mut ttft = None;
                let th = stream_chat(&agent(5, 300), &ep.url, ep.key.as_deref(), &ep.model, body, rank::strip_mode(&c, strip_thinking.as_deref(), heretic), |t| { ttft.get_or_insert(t0.elapsed().as_millis()); ans.push_str(t); if !js { print!("{t}"); let _ = so.flush(); } })?;
                eprintln!("\n\x1b[2m[Antwort {}: erstes Token {} ms, gesamt {} ms]\x1b[0m", ep.model, ttft.unwrap_or(0), t0.elapsed().as_millis());
                if show_thinking && !th.is_empty() { eprintln!("\x1b[2m{th}\x1b[0m\n"); }
                if js { println!("{}", json!({"answer": ans.trim(), "terms": f.terms, "passages": cs.iter().map(cj).collect::<Vec<_>>()})); }
                else { println!("\n\nQuellen:\n  {}", srcs()); }
            }
        }
        Cmd::Web { port } => web::run(&dir, &model, port)?,
        Cmd::Serve => {
            use std::io::BufRead;
            let (db, works) = open(&dir, &model)?;
            let mut so = std::io::stdout();
            for l in std::io::stdin().lock().lines() {
                let l = l?;
                if l.trim().is_empty() { continue; }
                let t = std::time::Instant::now();
                let r = serde_json::from_str::<Value>(&l).map_err(anyhow::Error::from).and_then(|v| {
                    let q = v["q"].as_str().ok_or_else(|| anyhow::anyhow!("missing q"))?;
                    Ok(cites(&db, &works, q, v["k"].as_u64().unwrap_or(3) as usize)?.iter().map(cj).collect::<Vec<_>>())
                });
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let o = match r { Ok(h) => json!({"hits": h, "ms": ms}), Err(e) => json!({"error": e.to_string(), "hits": [], "ms": ms}) };
                writeln!(so, "{o}")?; so.flush()?;
            }
        }
        Cmd::Dense { model_dir, out } => {
            let out = out.unwrap_or_else(|| dir.join("dense"));
            let wrap: Vec<String> = std::env::var("BUCH_GPU_WRAP").unwrap_or_default().split_whitespace().map(String::from).collect();
            if let (Some((w, a)), None) = (wrap.split_first(), std::env::var_os("BUCH_WRAPPED")) {
                let mut cmd = Command::new(w);
                cmd.args(a).arg(std::env::current_exe()?).arg("--dir").arg(&dir).arg("dense").arg("--out").arg(&out).env("BUCH_WRAPPED", "1");
                if let Some(m) = model_dir { cmd.arg(m); }
                if !cmd.status()?.success() { bail!("embedding failed"); }
                return Ok(());
            }
            let (db, _) = open(&dir, &model)?;
            let texts: Vec<String> = (0..db.npass).map(|pid| { let (_, t) = db.store.get(pid).unwrap_or_default(); format!("{}. {t}", db.art_of(pid).title.split(": ").next().unwrap_or("")) }).collect();
            dense::embed_passages(&texts, &out, model_dir)?;
        }
        Cmd::Dump { out } => {
            let (db, _) = open(&dir, &model)?;
            let mut w = std::io::BufWriter::new(fs::File::create(out)?);
            for pid in 0..db.npass { if let Some((sec, t)) = db.store.get(pid) { writeln!(w, "{}", json!({"pid": pid, "title": db.art_of(pid).title, "page": sec, "text": t}))?; } }
        }
        Cmd::Eval { file, expand: ex_on, rerank, ncand, ntop, legacy, retr } => {
            if let Some(r) = retr { unsafe { std::env::set_var("BUCH_RETR", r); } }
            if legacy { unsafe { std::env::set_var("WIKI_TITLEBOOST", "2.0"); std::env::remove_var("WIKI_NOCANDS"); } }
            let c = web::cfg();
            let (db, works) = open(&dir, &model)?;
            if legacy { unsafe { std::env::remove_var("WIKI_NOCANDS"); } }
            let qs: Vec<Vec<String>> = fs::read_to_string(file)?.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| l.split('\t').map(String::from).collect()).collect();
            let (mut h, mut ctx, mut pg, mut npg) = ([0usize; 4], 0, 0, 0);
            let mut ms = [0u128; 3];
            for q in &qs {
                let f = find(&db, &works, &c, &q[0], ex_on, &rerank, ncand, ntop, &mut |_, _| {})?;
                for w in &f.warn { eprintln!("warn: {w}"); }
                for i in 0..3 { ms[i] += f.ms[i]; }
                let want = q[1].to_lowercase();
                let r = f.top.iter().position(|c| c.title.to_lowercase().contains(&want));
                for (i, n) in [1, 3, 6, 10].iter().enumerate() { h[i] += r.is_some_and(|x| x < *n) as usize; }
                ctx += r.is_some() as usize;
                let mut pgs = String::new();
                if let Some(p) = q.get(2) {
                    npg += 1;
                    let ok = f.top.iter().any(|c| c.title.to_lowercase().contains(&want) && p.split('|').any(|x| x == c.page));
                    pg += ok as usize; pgs = format!(" page:{}", if ok { "ok" } else { "MISS" });
                }
                if let Some(t) = &f.terms { println!("     terms: {t}"); }
                println!("{} {:?}{}  {}  -> {}", if r.is_some_and(|x| x < 3) { "ok  " } else { "MISS" }, r.map(|x| x + 1), pgs, q[0], f.top.iter().take(3).map(|c| c.citation.as_str()).collect::<Vec<_>>().join(" | "));
            }
            let n = qs.len();
            println!("hit@1 {}/{n}  hit@3 {}/{n}  hit@6 {}/{n}  hit@10 {}/{n}  ctx(top{ntop}) {ctx}/{n}  page-in-ctx {pg}/{npg}", h[0], h[1], h[2], h[3]);
            println!("avg ms/question: expand {}  search {}  rerank {}  (rerank={rerank} ncand={ncand} expand={ex_on})", ms[0] / n as u128, ms[1] / n as u128, ms[2] / n as u128);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(t: &str) -> Cite { Cite { citation: t.into(), title: t.into(), page: "1".into(), url: "".into(), license: "".into(), score: 0.0, text: "".into(), author: "".into(), year: Value::Null, translator: Value::Null } }
    #[test] fn rrf_fuses_and_dedupes() {
        let r = rrf(vec![(1.0, vec![c("a"), c("b"), c("c")]), (1.0, vec![c("c"), c("d")])], 3);
        let n: Vec<_> = r.iter().map(|x| x.citation.as_str()).collect();
        assert_eq!(n, ["c", "a", "b"]);
    }
    #[test] fn terms_stop_at_chatter() {
        assert_eq!(clean_terms("Steuern Monopol 1789 Antwort nur Begriffe").as_deref(), Some("Steuern Monopol"));
        assert_eq!(clean_terms(&"wort ".repeat(30)).unwrap().split(' ').count(), 12);
    }
    #[test] fn terms_clean() {
        assert_eq!(clean_terms("\n Privatisierung, Verteidigung; Schutz-agentur.\nrest").as_deref(), Some("Privatisierung Verteidigung Schutz-agentur"));
        assert_eq!(clean_terms("  \n"), None);
    }
}
