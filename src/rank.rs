//! Model endpoints (auto-started local servers) and the rerank stage.
use super::{agent, stream_chat, think, web, Cite};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::process::Command;

#[derive(Clone)]
pub struct Ep { pub url: String, pub model: String, pub key: Option<String> }

fn up(url: &str) -> bool { agent(1, 2).get(&format!("{url}/health")).call().is_ok_and(|r| r.status().is_success()) }

/// port of a local helper server: `$BUCH_PORTS name` (optional port registry command) if it answers, else fixed default
pub fn port_of(name: &str) -> u16 {
    let def = match name { "buch-chat" => 8102, "buch-rerank" => 8103, _ => 8101 };
    let Some(reg) = std::env::var("BUCH_PORTS").ok().filter(|s| !s.is_empty()) else { return def };
    Command::new(reg).args(["port", name]).output().ok().filter(|o| o.status.success()).and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok()).unwrap_or(def)
}

/// base URL of a local helper server (`buch-chat` | `buch-rerank`); starts it via its script when down (<= 25 s)
pub fn base(name: &str) -> Option<String> {
    let url = format!("http://127.0.0.1:{}", port_of(name));
    if up(&url) { return Some(url); }
    Command::new(super::ex(&format!("~/.local/bin/{name}"))).arg("start").output().ok()?;
    up(&url).then_some(url)
}

/// fast local chat model (LFM3-1.7B)
pub fn fast() -> Result<Ep> {
    let b = base("buch-chat").context("LFM3-Server (buch-chat) nicht erreichbar")?;
    Ok(Ep { url: format!("{b}/v1"), model: "lfm3".into(), key: None })
}

/// the configurable large model (config.json endpoint/model/key), any OpenAI-compatible server (LM Studio, llama.cpp, mlx_lm, OpenRouter)
pub fn big(c: &Value) -> Ep {
    let m = web::s(c, "model");
    Ep { url: web::endpoint(c), model: if m.is_empty() { "local".into() } else { m.into() }, key: web::key(c) }
}

pub fn pick(env: &str, key: &str, c: &Value, def: &str) -> String {
    std::env::var(env).ok().filter(|s| !s.is_empty()).or_else(|| Some(web::s(c, key).to_string()).filter(|s| !s.is_empty())).unwrap_or(def.into())
}

/// answer model: "big" (default, Gemma via config endpoint) | "lfm3" (unreliable: misattributes, invents)
pub fn answer_ep(c: &Value) -> Result<Ep> { answer_ep_for(&pick("BUCH_ANSWER", "answer", c, "big"), c, || {}) }

/// endpoint for an answer kind (big | lfm3 | heretic); `wait` runs once before blocking on a cold heretic start
pub fn answer_ep_for(kind: &str, c: &Value, wait: impl FnOnce()) -> Result<Ep> {
    match kind { "lfm3" => fast(), "heretic" => heretic(wait), _ => Ok(big(c)) }
}

pub fn heretic_model() -> String { std::env::var("BUCH_LLM_MODEL").unwrap_or_else(|_| super::ex("~/AI-Model/models/zecanard/gemma-4-E4B-heretic-nothink").display().to_string()) }

/// test model Gemma-4 E4B heretic (mlx_lm.server via `buch-llm`); auto-starts, waits <= 60 s
pub fn heretic(wait: impl FnOnce()) -> Result<Ep> {
    let port = port_of("buch-llm");
    let url = format!("http://127.0.0.1:{port}/v1");
    let ok = || agent(1, 2).get(&format!("{url}/models")).call().is_ok_and(|r| r.status().is_success());
    if !ok() {
        wait();
        let _ = Command::new(super::ex("~/.local/bin/buch-llm")).arg("start").output();
        for _ in 0..60 { if ok() { break; } std::thread::sleep(std::time::Duration::from_secs(1)); }
        if !ok() { bail!("heretic-Server (buch-llm) nicht erreichbar"); }
    }
    Ok(Ep { url, model: heretic_model(), key: None })
}

/// the heretic model leaks reasoning before `<channel|>`: force stripping
pub fn strip_mode(c: &Value, flag: Option<&str>, heretic: bool) -> think::Mode { if heretic { think::Mode::Strict } else { think::Mode::resolve(flag, c) } }

/// LFM3 query expansion: off by default (noisy; costs ~1.7 s); BUCH_EXPAND=on or config "expand":"on"
pub fn expand_on(c: &Value) -> bool { pick("BUCH_EXPAND", "expand", c, "off") == "on" }

pub fn rerank_kind(c: &Value) -> String { pick("BUCH_RERANK", "rerank", c, "none") }

fn chat_once(ep: &Ep, body: Value, mode: think::Mode, secs: u64) -> Result<String> {
    let mut out = String::new();
    stream_chat(&agent(3, secs), &ep.url, ep.key.as_deref(), &ep.model, body, mode, |t| out.push_str(t))?;
    Ok(out)
}

fn clip(s: &str, n: usize) -> String { if s.chars().count() <= n { s.into() } else { format!("{} …", s.chars().take(n).collect::<String>().trim_end()) } }

/// text of a record without the "# title\n# S. n" header
fn body(c: &Cite) -> &str { c.text.splitn(3, '\n').nth(2).unwrap_or(&c.text) }

fn d1_scores(q: &str, cs: &[Cite]) -> Result<Vec<f32>> {
    let b = base("buch-rerank").context("d1-Server (buch-rerank) nicht erreichbar")?;
    let mut r = agent(3, 180).post(&format!("{b}/rank")).send_json(&json!({"query": q, "passages": cs.iter().map(|c| body(c)).collect::<Vec<_>>(), "mode": "noul"}))?;
    let v: Value = r.body_mut().read_json()?;
    let s: Vec<f32> = v["scores"].as_array().context(v["error"].to_string())?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
    if s.len() != cs.len() { bail!("d1: {} scores for {} passages", s.len(), cs.len()); }
    Ok(s)
}

/// P(Ja) from the first-token logprobs of the pointwise yes/no prompt
fn yes_prob(ep: &Ep, q: &str, c: &Cite) -> Result<f32> {
    let u = format!("Frage: {q}\n\nTextstelle:\n{}\n\nEnthält die Textstelle Aussagen, die zur Beantwortung der Frage direkt beitragen? Die Wortwahl kann von der Frage abweichen. Antworte nur mit Ja oder Nein.", clip(body(c), 1500));
    let req = json!({"model": ep.model, "temperature": 0, "max_tokens": 1, "logprobs": true, "top_logprobs": 12,
        "messages": [{"role": "system", "content": "Du bewertest Suchtreffer für eine Bibliothek."}, {"role": "user", "content": u}]});
    let mut r = agent(3, 60).post(&format!("{}/chat/completions", ep.url)).send_json(&req)?;
    let v: Value = r.body_mut().read_json()?;
    let tl = v["choices"][0]["logprobs"]["content"][0]["top_logprobs"].as_array().context("no logprobs")?;
    let (mut y, mut n) = (0f32, 0f32);
    for t in tl {
        let (s, p) = (t["token"].as_str().unwrap_or("").trim().to_lowercase(), (t["logprob"].as_f64().unwrap_or(-99.0) as f32).exp());
        if s.starts_with("ja") { y += p } else if s.starts_with("nein") { n += p }
    }
    Ok(y / (y + n).max(1e-6))
}

fn point_scores(q: &str, cs: &[Cite]) -> Result<Vec<f32>> {
    let ep = fast()?;
    let mut out = vec![0f32; cs.len()];
    let err = std::sync::Mutex::new(None);
    std::thread::scope(|sc| {
        for (ch, o) in cs.chunks(10).zip(out.chunks_mut(10)) {
            let (ep, err) = (&ep, &err);
            sc.spawn(move || for (c, o) in ch.iter().zip(o.iter_mut()) { match yes_prob(ep, q, c) { Ok(p) => *o = p, Err(e) => { *err.lock().unwrap() = Some(e); return } } });
        }
    });
    if let Some(e) = err.into_inner().unwrap() { return Err(e); }
    Ok(out)
}

/// listwise: one prompt with excerpts, model names the best `n` numbers; returns scores that encode that order
fn list_scores(ep: &Ep, q: &str, cs: &[Cite], n: usize, mode: think::Mode) -> Result<Vec<f32>> {
    let list: String = cs.iter().enumerate().map(|(i, c)| format!("[{}] {}\n", i + 1, clip(body(c), 380))).collect();
    let u = format!("Frage: {q}\n\nTextstellen:\n{list}\nNenne die Nummern der {n} Textstellen, die die Frage am besten beantworten oder belegen, wichtigste zuerst. Die Wortwahl kann von der Frage abweichen. Antworte nur mit den Nummern, durch Komma getrennt.");
    let req = json!({"temperature": 0, "max_tokens": 80, "messages": [{"role": "system", "content": "Du sortierst Suchtreffer nach Relevanz für eine Frage."}, {"role": "user", "content": u}]});
    let o = chat_once(ep, req, mode, 120)?;
    let mut seen = vec![];
    for t in o.split(|ch: char| !ch.is_ascii_digit()).filter(|t| !t.is_empty()) {
        if let Ok(i) = t.parse::<usize>() { if i >= 1 && i <= cs.len() && !seen.contains(&(i - 1)) { seen.push(i - 1); } }
    }
    if seen.is_empty() { bail!("listwise: no numbers in {o:?}"); }
    let mut s = vec![0f32; cs.len()];
    for (r, i) in seen.iter().enumerate() { s[*i] = 1.0 - r as f32 * 0.01; }
    Ok(s)
}

/// kinds: none | d1 | lfm3 (pointwise) | lfm3list | big (listwise with the configured large model)
pub fn rerank(c: &Value, kind: &str, q: &str, cands: &[Cite], n: usize) -> Result<Vec<Cite>> {
    if kind == "none" { return Ok(cands.iter().take(n).cloned().collect()); }
    let s = match kind {
        "d1" => d1_scores(q, cands)?,
        "lfm3" => point_scores(q, cands)?,
        "lfm3list" => list_scores(&fast()?, q, cands, n, think::Mode::Off)?,
        "big" => list_scores(&big(c), q, cands, n, think::Mode::resolve(None, c))?,
        k => bail!("unknown rerank kind {k}"),
    };
    let mut ix: Vec<usize> = (0..cands.len()).collect();
    ix.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
    Ok(ix.into_iter().take(n).map(|i| { let mut c = cands[i].clone(); c.score = s[i]; c }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(n: &str) -> Cite { Cite { citation: n.into(), title: n.into(), page: "1".into(), url: "".into(), license: "".into(), score: 0.0, text: format!("# t\n# S. 1\n{n} text"), author: "".into(), year: Value::Null, translator: Value::Null } }
    #[test] fn none_keeps_fused_order() {
        let r = rerank(&json!({}), "none", "q", &[c("a"), c("b"), c("c")], 2).unwrap();
        assert_eq!(r.iter().map(|x| x.citation.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }
    #[test] fn unknown_kind_errors() { assert!(rerank(&json!({}), "xyz", "q", &[c("a"), c("b")], 1).is_err()); }
    #[test] fn body_strips_header() { assert_eq!(body(&c("a")), "a text"); }
    #[test] fn clip_cuts() { assert_eq!(clip("abcdef", 3), "abc …"); assert_eq!(clip("ab", 3), "ab"); }
    #[test] fn pick_prefers_config() { assert_eq!(pick("BUCH_NO_SUCH_ENV", "k", &json!({"k": " x "}), "d"), "x"); assert_eq!(pick("BUCH_NO_SUCH_ENV", "k", &json!({}), "d"), "d"); }
}
