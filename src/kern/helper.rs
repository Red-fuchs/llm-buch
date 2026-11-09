use anyhow::{Context, Result};
use std::{io::{Read, Write}, os::unix::net::UnixStream, path::PathBuf, time::{Duration, Instant}};

pub const D: usize = 384;

/// Client for the resident e5 helper (embd/serve.py): one query line in, 384 f32 LE out. Tries each socket; spawns at the last one.
pub struct Helper { socks: Vec<PathBuf> }

fn one(p: &PathBuf, q: &str) -> Result<Vec<f32>> {
    let mut s = UnixStream::connect(p)?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    s.write_all(q.replace('\n', " ").as_bytes())?; s.write_all(b"\n")?;
    let mut b = vec![0u8; D * 4];
    s.read_exact(&mut b)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl Helper {
    pub fn new(socks: Vec<PathBuf>) -> Self { Self { socks } }

    fn rpc(&self, q: &str) -> Result<Vec<f32>> {
        let mut err = anyhow::anyhow!("no socket");
        for p in &self.socks { match one(p, q) { Ok(v) => return Ok(v), Err(e) => err = e } }
        Err(err)
    }

    fn spawn(&self) -> Result<std::process::Child> {
        use std::os::unix::process::CommandExt;
        let sock = self.socks.last().unwrap();
        let home = std::env::var("HOME").unwrap_or_default();
        let script = std::env::var("BUCH_EMBD").unwrap_or(format!("{home}/.local/share/buch/serve.py"));
        let py = crate::py();
        let log = std::fs::File::create(sock.with_extension("log"))?;
        Ok(std::process::Command::new(py).arg("-I").arg(&script).arg(sock).env("HF_HUB_OFFLINE", "1")
            .stdin(std::process::Stdio::null()).stdout(log.try_clone()?).stderr(log).process_group(0).spawn()?)
    }

    /// Embed via the helper; starts it detached (idle-exit) when down. `wait=false` returns an error right after starting it.
    pub fn embed(&self, q: &str, wait: bool) -> Result<Vec<f32>> {
        if let Ok(v) = self.rpc(q) { return Ok(v); }
        let mut ch = self.spawn().context("embd helper unavailable (cannot start python)")?;
        if !wait { anyhow::bail!("embd helper not running, starting it"); }
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(150));
            if let Ok(v) = self.rpc(q) { return Ok(v); }
            if ch.try_wait()?.is_some() { break; }
        }
        anyhow::bail!("embd helper unavailable")
    }
}
