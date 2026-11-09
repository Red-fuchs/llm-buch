use anyhow::Result;
use serde_json::{json, Value};
use std::{io::{BufRead, BufReader, Read}, path::{Path, PathBuf}, sync::{mpsc, Arc}, time::Duration};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use crate::kern::db::Db;

const INDEX: &str = include_str!("index.html");

struct St { db: Db, works: Value }

struct Chan { rx: mpsc::Receiver<Vec<u8>>, buf: Vec<u8>, pos: usize }
impl Read for Chan {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            match self.rx.recv() { Ok(b) => { self.buf = b; self.pos = 0; } Err(_) => return Ok(0) }
        }
        let n = out.len().min(self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn cfg_path() -> PathBuf { super::ex("~/.config/buch/config.json") }
pub fn cfg() -> Value { std::fs::read(cfg_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({})) }
pub fn s<'a>(v: &'a Value, k: &str) -> &'a str { v[k].as_str().unwrap_or("").trim() }

pub fn endpoint(c: &Value) -> String {
    let e = s(c, "endpoint");
    if !e.is_empty() { return e.into(); }
    std::env::var("BUCH_ENDPOINT").unwrap_or("http://localhost:1234/v1".into())
}
pub fn key(c: &Value) -> Option<String> {
    let k = s(c, "key");
    if !k.is_empty() { return Some(k.into()); }
    std::env::var("BUCH_KEY").ok().filter(|k| !k.is_empty())
}

fn agent(secs: u64) -> ureq::Agent { ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(secs))).http_status_as_error(false).build().into() }

/// model ids of the endpoint, or the error text
fn models(ep: &str, key: &Option<String>) -> Result<Vec<String>, String> {
    let mut r = agent(4).get(&format!("{}/models", ep.trim_end_matches('/')));
    if let Some(k) = key { r = r.header("Authorization", format!("Bearer {k}")); }
    let mut resp = r.call().map_err(|e| e.to_string())?;
    if !resp.status().is_success() { return Err(format!("HTTP {}", resp.status())); }
    let v: Value = resp.body_mut().read_json().map_err(|e| e.to_string())?;
    Ok(v["data"].as_array().map(|a| a.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect()).unwrap_or_default())
}

fn hdr(k: &str, v: &str) -> Header { Header::from_bytes(k, v).unwrap() }
fn jr(code: u16, v: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(v.to_string().into_bytes()).with_status_code(code).with_header(hdr("Content-Type", "application/json; charset=utf-8")).with_header(hdr("Cache-Control", "no-store"))
}

fn hint(k: &str) -> String { if k.len() > 8 { format!("…{}", &k[k.len() - 4..]) } else if k.is_empty() { "".into() } else { "…".into() } }

fn sse(tx: &mpsc::Sender<Vec<u8>>, ev: &str, d: &Value) -> bool { tx.send(format!("event: {ev}\ndata: {d}\n\n").into_bytes()).is_ok() }

fn ask(st: Arc<St>, q: String, k: usize, tx: mpsc::Sender<Vec<u8>>) {
    let c = cfg();
    let kind = super::rank::rerank_kind(&c);
    let t0 = std::time::Instant::now();
    let mut alive = true;
    let f = match super::find(&st.db, &st.works, &c, &q, super::rank::expand_on(&c), &kind, 30, k, &mut |n, v| { if !sse(&tx, n, &v) { alive = false } }) { Ok(f) => f, Err(e) => { sse(&tx, "error", &json!(e.to_string())); return } };
    if !alive { return; }
    for w in &f.warn { sse(&tx, "warn", &json!(w)); }
    let cs = &f.top;
    if !sse(&tx, "passages", &Value::Array(cs.iter().map(super::cj).collect())) { return; }
    let akind = super::rank::pick("BUCH_ANSWER", "answer", &c, "big");
    let ep = match super::rank::answer_ep_for(&akind, &c, || { sse(&tx, "status", &json!("Lade Modell …")); }) { Ok(e) => e, Err(e) => { sse(&tx, "nollm", &json!(e.to_string())); sse(&tx, "done", &json!(true)); return } };
    let ky = ep.key.clone();
    if models(&ep.url, &ky).is_err() && ep.key.is_none() && ep.model != "lfm3" && akind != "heretic" { sse(&tx, "nollm", &json!("Modell-Server nicht erreichbar")); sse(&tx, "done", &json!(true)); return; }
    let model = if ep.model == "lfm3" || akind == "heretic" || !s(&c, "model").is_empty() { ep.model.clone() } else { models(&ep.url, &ky).ok().and_then(|m| m.first().cloned()).unwrap_or("local".into()) };
    sse(&tx, "status", &json!("Schreibe Antwort …"));
    let body = super::answer_body(&model, &q, cs, true);
    let ep_url = ep.url.clone();
    let mut r = agent(300).post(&format!("{}/chat/completions", ep_url.trim_end_matches('/')));
    if let Some(k) = &ky { r = r.header("Authorization", format!("Bearer {k}")); }
    let resp = match r.send_json(&body) { Ok(r) => r, Err(e) => { sse(&tx, "error", &json!(e.to_string())); return } };
    if !resp.status().is_success() { sse(&tx, "error", &json!(format!("Modell-Server: HTTP {}", resp.status()))); return; }
    sse(&tx, "timing", &json!({"expand": f.ms[0], "search": f.ms[1], "rerank": f.ms[2], "model": model, "rerank_kind": kind, "before_answer": t0.elapsed().as_millis() as u64}));
    let mode = super::rank::strip_mode(&c, std::env::var("BUCH_STRIP_THINKING").ok().as_deref(), akind == "heretic");
    let (mut sp, mut told) = (super::think::Splitter::new(mode, 0, 2000), false);
    for l in BufReader::new(resp.into_body().into_reader()).lines() {
        let Ok(l) = l else { break };
        let Some(d) = l.strip_prefix("data:") else { continue };
        let d = d.trim();
        if d == "[DONE]" { break; }
        if let Ok(v) = serde_json::from_str::<Value>(d) {
            let dl = &v["choices"][0]["delta"];
            if let Some(t) = dl["reasoning_content"].as_str().or(dl["reasoning"].as_str()).filter(|t| !t.is_empty()) {
                sp.reasoning(t);
                if !told { told = true; if !sse(&tx, "thinking", &json!(true)) { return; } }
            }
            if let Some(t) = dl["content"].as_str() {
                if t.is_empty() { continue; }
                if sp.is_thinking() && !told { told = true; if !sse(&tx, "thinking", &json!(true)) { return; } }
                let o = sp.push(t);
                if sp.take_reset() && !sse(&tx, "reset", &json!(true)) { return; }
                if !o.is_empty() && !sse(&tx, "token", &json!(o)) { return; }
            }
        }
    }
    let o = sp.finish();
    if !o.is_empty() { sse(&tx, "token", &json!(o)); }
    sse(&tx, "done", &json!(true));
}

fn handle(st: Arc<St>, mut req: Request) {
    let h = |n: &str| req.headers().iter().find(|x| x.field.as_str().as_str().eq_ignore_ascii_case(n)).map(|x| x.value.as_str().to_string());
    let host_ok = h("Host").is_some_and(|v| { let v = v.split(':').next().unwrap_or(""); v == "127.0.0.1" || v == "localhost" });
    let origin_ok = h("Origin").is_none_or(|o| o.starts_with("http://127.0.0.1") || o.starts_with("http://localhost"));
    if !host_ok || !origin_ok { let _ = req.respond(jr(403, json!({"error": "forbidden"}))); return; }
    let (m, url) = (req.method().clone(), req.url().split('?').next().unwrap_or("/").to_string());
    let mut body = String::new();
    if m == Method::Post { let _ = req.as_reader().take(1 << 20).read_to_string(&mut body); }
    let b: Value = serde_json::from_str(&body).unwrap_or(json!({}));
    let _ = match (&m, url.as_str()) {
        (Method::Get, "/") => req.respond(Response::from_string(INDEX).with_header(hdr("Content-Type", "text/html; charset=utf-8"))),
        (Method::Get, "/api/books") => {
            let mut v: Vec<Value> = st.works.as_object().map(|o| o.iter().map(|(t, w)| { let mut w = w.clone(); w["title"] = json!(t); w }).collect()).unwrap_or_default();
            v.sort_by_key(|w| (w["author"].as_str().unwrap_or("").to_string(), w["work"].as_str().unwrap_or("").to_string()));
            req.respond(jr(200, json!(v)))
        }
        (Method::Get, "/api/config") => {
            let c = cfg();
            let k = key(&c).unwrap_or_default();
            req.respond(jr(200, json!({"endpoint": endpoint(&c), "model": s(&c, "model"), "answer": super::rank::pick("BUCH_ANSWER", "answer", &c, "big"), "rerank": super::rank::rerank_kind(&c), "key_set": !k.is_empty(), "key_hint": hint(&k)})))
        }
        (Method::Post, "/api/config") => {
            let mut c = cfg();
            if let Some(e) = b["endpoint"].as_str() { c["endpoint"] = json!(e.trim()); }
            if let Some(e) = b["model"].as_str() { c["model"] = json!(e.trim()); }
            if let Some(e) = b["answer"].as_str().filter(|e| ["lfm3", "big", "heretic"].contains(e)) { c["answer"] = json!(e); }
            if let Some(e) = b["rerank"].as_str().filter(|e| ["none", "d1", "lfm3", "lfm3list", "big"].contains(e)) { c["rerank"] = json!(e); }
            if b["clear_key"].as_bool() == Some(true) { c["key"] = json!(""); }
            else if let Some(k) = b["key"].as_str().filter(|k| !k.trim().is_empty()) { c["key"] = json!(k.trim()); }
            let p = cfg_path();
            let r = (|| -> std::io::Result<()> {
                use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
                std::fs::create_dir_all(p.parent().unwrap())?;
                std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&p).and_then(|mut f| std::io::Write::write_all(&mut f, serde_json::to_string_pretty(&c)?.as_bytes()))?;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600))
            })();
            match r { Ok(_) => req.respond(jr(200, json!({"ok": true}))), Err(e) => req.respond(jr(500, json!({"error": e.to_string()}))) }
        }
        (Method::Post, "/api/test") => {
            let c = cfg();
            let ep = b["endpoint"].as_str().map(|e| e.trim().to_string()).filter(|e| !e.is_empty()).unwrap_or(endpoint(&c));
            let ky = b["key"].as_str().filter(|k| !k.trim().is_empty()).map(String::from).or(key(&c));
            req.respond(jr(200, match models(&ep, &ky) { Ok(m) => json!({"ok": true, "models": m}), Err(e) => json!({"ok": false, "error": e}) }))
        }
        (Method::Post, "/api/ask") => {
            let q = b["q"].as_str().unwrap_or("").trim().to_string();
            if q.is_empty() { req.respond(jr(400, json!({"error": "missing q"}))) }
            else {
                let k = b["k"].as_u64().unwrap_or(10).clamp(1, 12) as usize;
                let (tx, rx) = mpsc::channel();
                let s2 = st.clone();
                std::thread::spawn(move || ask(s2, q, k, tx));
                req.respond(Response::new(StatusCode(200), vec![hdr("Content-Type", "text/event-stream"), hdr("Cache-Control", "no-store")], Chan { rx, buf: vec![], pos: 0 }, None, None))
            }
        }
        _ => req.respond(jr(404, json!({"error": "not found"}))),
    };
}

pub fn run(dir: &Path, model: &Path, port: u16) -> Result<()> {
    let (db, works) = super::open(dir, model)?;
    let st = Arc::new(St { db, works });
    let srv = Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("bind 127.0.0.1:{port}: {e}"))?;
    eprintln!("buch web: http://127.0.0.1:{port}/");
    for req in srv.incoming_requests() { let st = st.clone(); std::thread::spawn(move || handle(st, req)); }
    Ok(())
}
