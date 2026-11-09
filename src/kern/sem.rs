use anyhow::Result;
use memmap2::Mmap;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const D: usize = 384;
const BB: usize = D / 8;
const QB: usize = D / 2;

/// Lead-passage semantic layer: e5-small vectors, one per article (row = article id, or via lead.ids).
/// lead.bin = sign bits of (v - mean), lead.q4 = int4 (offset 8, all-zero row = no lead). Query vectors come from the resident helper embd/serve.py.
pub struct Sem { bin: Mmap, q4: Mmap, ids: Option<Mmap>, mean: Vec<f32>, s4: f32 }

fn mm(p: &Path) -> Result<Mmap> { Ok(unsafe { Mmap::map(&std::fs::File::open(p)?)? }) }

impl Sem {
    pub fn open(dir: &Path) -> Option<Self> {
        let d = std::env::var("WIKI_LEAD").map(PathBuf::from).unwrap_or(dir.into());
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("lead.meta.json")).ok()?).ok()?;
        Some(Self { bin: mm(&d.join("lead.bin")).ok()?, q4: mm(&d.join("lead.q4")).ok()?, ids: mm(&d.join("lead.ids")).ok(),
            mean: serde_json::from_value(meta["mean"].clone()).ok()?, s4: meta["s4"].as_f64()? as f32 })
    }

    fn embed(&self, q: &str) -> Result<Vec<f32>> { crate::dense::query_vec(q) }

    /// Top-n (article id, cosine) over all leads: Hamming scan -> top 200 -> int4 dot.
    pub fn search(&self, q: &str, n: usize) -> Result<Vec<(usize, f32)>> {
        let qv = self.embed(q)?;
        let mut qb = [0u8; BB];
        for (i, (x, m)) in qv.iter().zip(&self.mean).enumerate() { if x - m > 0.0 { qb[i / 8] |= 1 << (i % 8); } }
        let qw: Vec<u64> = qb.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let words: &[u64] = unsafe { std::slice::from_raw_parts(self.bin.as_ptr() as *const u64, self.bin.len() / 8) };
        const CH: usize = 1 << 16;
        let w = BB / 8;
        let mut cand: Vec<(u32, u32)> = words.par_chunks(CH * w).enumerate().flat_map(|(ci, c)| {
            let mut v: Vec<(u32, u32)> = c.chunks_exact(w).enumerate().map(|(i, p)| (p.iter().zip(&qw).map(|(a, b)| (a ^ b).count_ones()).sum(), (ci * CH + i) as u32)).collect();
            if v.len() > 200 { v.select_nth_unstable(200); v.truncate(200); }
            v
        }).collect();
        if cand.len() > 200 { cand.select_nth_unstable(200); cand.truncate(200); }
        let mut r: Vec<(usize, f32)> = cand.iter().filter_map(|&(_, row)| {
            let row = row as usize;
            let b = &self.q4[row * QB..][..QB];
            if b.iter().all(|&x| x == 0) { return None; }
            let s: f32 = b.iter().enumerate().map(|(j, &x)| ((x & 15) as f32 - 8.0) * qv[2 * j] + ((x >> 4) as f32 - 8.0) * qv[2 * j + 1]).sum::<f32>() / self.s4;
            let id = match &self.ids { Some(m) => u32::from_le_bytes(m[row * 4..][..4].try_into().unwrap()) as usize, None => row };
            Some((id, s))
        }).collect();
        r.sort_by(|a, b| b.1.total_cmp(&a.1));
        r.truncate(n);
        Ok(r)
    }
}
