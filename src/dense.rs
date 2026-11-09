//! Dense retrieval: multilingual-e5-small in-process (candle, CPU; Accelerate on macOS). Vectors: f16, L2-normalised, row = passage id.
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use memmap2::Mmap;
use std::{io::{Read, Write}, path::{Path, PathBuf}, sync::{Mutex, OnceLock}, time::{Duration, Instant}};
use tokenizers::Tokenizer;

pub const REPO: &str = "intfloat/multilingual-e5-small";
const FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

pub struct Embedder { m: BertModel, tok: Tokenizer, dim: usize }

impl Embedder {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &Device::Cpu)? };
        let dim = cfg.hidden_size;
        Ok(Self { m: BertModel::load(vb, &cfg)?, tok, dim })
    }

    /// L2-normalised mean-pooled vectors, row-major in input order. Texts need their "query: " / "passage: " prefix already.
    pub fn embed(&self, texts: &[String], maxlen: usize, bs: usize, mut prog: impl FnMut(usize)) -> Result<Vec<f32>> {
        let enc = self.tok.encode_batch(texts.to_vec(), true).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids: Vec<Vec<u32>> = enc.iter().map(|e| { let i = e.get_ids(); if i.len() > maxlen { let mut v = i[..maxlen - 1].to_vec(); v.push(2); v } else { i.to_vec() } }).collect();
        let mut ord: Vec<usize> = (0..ids.len()).collect();
        ord.sort_by_key(|&i| ids[i].len());
        let mut out = vec![0f32; ids.len() * self.dim];
        let dev = Device::Cpu;
        for (done, ch) in ord.chunks(bs.max(1)).enumerate() {
            let t = ch.iter().map(|&i| ids[i].len()).max().unwrap_or(1);
            let (mut a, mut m) = (vec![1u32; ch.len() * t], vec![0u32; ch.len() * t]);
            for (r, &i) in ch.iter().enumerate() { a[r * t..r * t + ids[i].len()].copy_from_slice(&ids[i]); m[r * t..r * t + ids[i].len()].fill(1); }
            let (a, m) = (Tensor::from_vec(a, (ch.len(), t), &dev)?, Tensor::from_vec(m, (ch.len(), t), &dev)?);
            let h = self.m.forward(&a, &a.zeros_like()?, Some(&m))?;
            let mf = m.to_dtype(DType::F32)?.unsqueeze(2)?;
            let p = h.broadcast_mul(&mf)?.sum(1)?.broadcast_div(&mf.sum(1)?)?;
            let p = p.broadcast_div(&p.sqr()?.sum_keepdim(1)?.sqrt()?)?;
            let v = p.flatten_all()?.to_vec1::<f32>()?;
            for (r, &i) in ch.iter().enumerate() { out[i * self.dim..][..self.dim].copy_from_slice(&v[r * self.dim..][..self.dim]); }
            prog((done + 1) * bs);
        }
        Ok(out)
    }
}

fn hub_dir() -> PathBuf {
    let ev = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty()).map(PathBuf::from);
    ev("HF_HUB_CACHE").or_else(|| ev("HF_HOME").map(|h| h.join("hub")))
        .or_else(|| ev("XDG_CACHE_HOME").map(|h| h.join("huggingface").join("hub")))
        .unwrap_or_else(|| crate::paths::home().join(".cache").join("huggingface").join("hub"))
}

fn complete(d: &Path) -> bool { FILES.iter().all(|f| d.join(f).exists()) }

/// model dir: BUCH_E5_DIR, HF cache (HF_HOME / HF_HUB_CACHE respected), else download (~470 MB; not with HF_HUB_OFFLINE=1)
pub fn model_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var("BUCH_E5_DIR") { let d = crate::paths::ex(&d); if complete(&d) { return Ok(d); } bail!("BUCH_E5_DIR {} lacks {FILES:?}", d.display()); }
    let repo = hub_dir().join(format!("models--{}", REPO.replace('/', "--")));
    if let Ok(rd) = std::fs::read_dir(repo.join("snapshots")) {
        let mut s: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| complete(p)).collect();
        s.sort();
        if let Some(p) = s.pop() { return Ok(p); }
    }
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| v == "1") { bail!("{REPO} not in the HF cache ({}) and HF_HUB_OFFLINE=1", hub_dir().display()); }
    let d = repo.join("snapshots").join("buch");
    std::fs::create_dir_all(&d)?;
    eprintln!("downloading {REPO} (~470 MB) to {}", d.display());
    for f in FILES { if !d.join(f).exists() { download(&format!("https://huggingface.co/{REPO}/resolve/main/{f}"), &d.join(f))?; } }
    Ok(d)
}

/// BUCH_DL_LIMIT = MB/s cap (default: unlimited)
fn download(url: &str, out: &Path) -> Result<()> {
    let lim: f64 = std::env::var("BUCH_DL_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0) * 1e6;
    let ag: ureq::Agent = ureq::Agent::config_builder().timeout_global(None).timeout_recv_body(None).build().into();
    let resp = ag.get(url).call().with_context(|| format!("download {url}"))?;
    if !resp.status().is_success() { bail!("download {url}: HTTP {}", resp.status()); }
    let tmp = out.with_extension("part");
    let mut f = std::fs::File::create(&tmp)?;
    let mut r = resp.into_body().into_reader();
    let (mut buf, mut n, t) = (vec![0u8; 1 << 16], 0u64, Instant::now());
    loop {
        let k = r.read(&mut buf)?;
        if k == 0 { break; }
        f.write_all(&buf[..k])?;
        n += k as u64;
        if lim > 0.0 { let want = Duration::from_secs_f64(n as f64 / lim); if let Some(w) = want.checked_sub(t.elapsed()) { std::thread::sleep(w); } }
    }
    drop(f);
    std::fs::rename(&tmp, out)?;
    Ok(())
}

static QE: OnceLock<std::result::Result<Mutex<Embedder>, String>> = OnceLock::new();

pub fn query_vec(q: &str) -> Result<Vec<f32>> {
    let e = QE.get_or_init(|| model_dir().and_then(|d| Embedder::load(&d)).map(Mutex::new).map_err(|e| format!("{e:#}")));
    let e = e.as_ref().map_err(|e| anyhow::anyhow!("dense embedder unavailable: {e}"))?;
    let e = e.lock().unwrap();
    e.embed(&[format!("query: {}", q.replace('\n', " "))], 64, 1, |_| {})
}

/// Embed `texts` (prefix "passage: " added) to f16 rows; writes <out>/dense.f16 + dense.json.
pub fn embed_passages(texts: &[String], out: &Path, model: Option<PathBuf>) -> Result<()> {
    let dir = match model { Some(m) => m, None => model_dir()? };
    let e = Embedder::load(&dir)?;
    let t0 = Instant::now();
    let tx: Vec<String> = texts.iter().map(|t| format!("passage: {t}")).collect();
    let v = e.embed(&tx, 512, 16, |n| eprint!("\r{}/{} passages, {:.0}s   ", n.min(tx.len()), tx.len(), t0.elapsed().as_secs_f32()))?;
    std::fs::create_dir_all(out)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(out.join("dense.f16"))?);
    for x in &v { w.write_all(&half::f16::from_f32(*x).to_le_bytes())?; }
    w.flush()?;
    std::fs::write(out.join("dense.json"), serde_json::to_vec(&serde_json::json!({"dim": e.dim, "model": REPO, "n": texts.len(), "backend": "candle"}))?)?;
    eprintln!("\n{} x {} in {:.0}s", texts.len(), e.dim, t0.elapsed().as_secs_f32());
    Ok(())
}

/// Dense passage vectors (f16, row = passage id); the query embedder runs in-process (loaded on first search).
pub struct Dense { v: Mmap, dim: usize, n: usize }

impl Dense {
    pub fn open(dir: &Path) -> Option<Self> {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("dense.json")).ok()?).ok()?;
        let v = unsafe { Mmap::map(&std::fs::File::open(dir.join("dense.f16")).ok()?).ok()? };
        let dim = m["dim"].as_u64()? as usize;
        Some(Self { n: v.len() / (2 * dim), v, dim })
    }

    /// top-n (passage id, cosine)
    pub fn search(&self, q: &str, n: usize) -> Result<Vec<(usize, f32)>> {
        let qv = query_vec(q)?;
        let mut r: Vec<(usize, f32)> = (0..self.n).map(|i| {
            let row = &self.v[i * self.dim * 2..][..self.dim * 2];
            (i, row.chunks_exact(2).zip(&qv).map(|(b, x)| half::f16::from_le_bytes([b[0], b[1]]).to_f32() * x).sum())
        }).collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        r.truncate(n);
        Ok(r)
    }
}
