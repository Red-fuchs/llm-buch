use anyhow::{bail, Result};
use memmap2::Mmap;
use std::{io::{Read, Write}, os::unix::net::UnixStream, path::{Path, PathBuf}, time::{Duration, Instant}};

/// Dense passage vectors (multilingual-e5, f16, row = passage id) + resident query embedder (embd/dense.py serve, idle exit).
pub struct Dense { v: Mmap, dim: usize, n: usize, sock: PathBuf, model: String }

impl Dense {
    pub fn open(dir: &Path) -> Option<Self> {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("dense.json")).ok()?).ok()?;
        let v = unsafe { Mmap::map(&std::fs::File::open(dir.join("dense.f16")).ok()?).ok()? };
        let dim = m["dim"].as_u64()? as usize;
        Some(Self { n: v.len() / (2 * dim), v, dim, sock: dir.join("dense.sock"), model: m["path"].as_str()?.into() })
    }

    fn rpc(&self, q: &str) -> Result<Vec<f32>> {
        let mut s = UnixStream::connect(&self.sock)?;
        s.set_read_timeout(Some(Duration::from_secs(10)))?;
        s.write_all(q.replace('\n', " ").as_bytes())?; s.write_all(b"\n")?;
        let mut b = vec![0u8; self.dim * 4];
        s.read_exact(&mut b)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn embed(&self, q: &str) -> Result<Vec<f32>> {
        if let Ok(v) = self.rpc(q) { return Ok(v); }
        let home = std::env::var("HOME").unwrap_or_default();
        let script = std::env::var("BUCH_DENSE_PY").unwrap_or(format!("{home}/.local/share/buch/dense.py"));
        let py = super::py();
        {
            use std::os::unix::process::CommandExt;
            let log = std::fs::File::create(self.sock.with_extension("log"))?;
            std::process::Command::new(py).arg("-I").arg(&script).arg("serve").arg(&self.model).arg(&self.sock).env("HF_HUB_OFFLINE", "1").env("DENSE_CPU", "1")
                .stdin(std::process::Stdio::null()).stdout(log.try_clone()?).stderr(log).process_group(0).spawn()?;
        }
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(25) {
            std::thread::sleep(Duration::from_millis(150));
            if let Ok(v) = self.rpc(q) { return Ok(v); }
        }
        bail!("dense embedder unavailable")
    }

    /// top-n (passage id, cosine)
    pub fn search(&self, q: &str, n: usize) -> Result<Vec<(usize, f32)>> {
        let qv = self.embed(q)?;
        let mut r: Vec<(usize, f32)> = (0..self.n).map(|i| {
            let row = &self.v[i * self.dim * 2..][..self.dim * 2];
            (i, row.chunks_exact(2).zip(&qv).map(|(b, x)| half_to_f32(u16::from_le_bytes([b[0], b[1]])) * x).sum())
        }).collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        r.truncate(n);
        Ok(r)
    }
}

fn half_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let b = match (e, m) {
        (0, 0) => s << 31,
        (0, _) => { let mut e = 113u32; let mut m = m; while m & 1024 == 0 { m <<= 1; e -= 1; } (s << 31) | (e << 23) | ((m & 1023) << 13) }
        (31, _) => (s << 31) | (255 << 23) | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(b)
}

#[cfg(test)]
mod tests {
    #[test] fn half() { assert_eq!(super::half_to_f32(0x3C00), 1.0); assert_eq!(super::half_to_f32(0xC000), -2.0); assert_eq!(super::half_to_f32(0), 0.0); }
}
