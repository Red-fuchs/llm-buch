use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::path::Path;
use tokenizers::Tokenizer;

pub const MAXTOK: usize = 256;

/// Model2Vec static embedder: tokenizer + embedding table, mean-pool.
pub struct Emb {
    tok: Tokenizer,
    mm: Mmap,
    off: usize,
    pub dim: usize,
    rows: usize,
    dt: Dtype,
    w: Option<Vec<f32>>,
    map: Option<Vec<u32>>,
}

impl Emb {
    pub fn load(dir: &Path) -> Result<Self> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
        let f = std::fs::File::open(dir.join("model.safetensors")).context("model.safetensors missing")?;
        let mm = unsafe { Mmap::map(&f)? };
        let st = SafeTensors::deserialize(&mm)?;
        let t = st.tensor("embeddings").or_else(|_| st.tensor("0")).context("no embeddings tensor")?;
        let (rows, dim) = (t.shape()[0], t.shape()[1]);
        let dt = t.dtype();
        let off = t.data().as_ptr() as usize - mm.as_ptr() as usize;
        let w = st.tensor("weights").ok().map(|t| t.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
        let map = st.tensor("mapping").ok().map(|t| t.data().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect());
        drop(st);
        Ok(Self { tok, mm, off, dim, rows, dt, w, map })
    }

    fn add(&self, id: usize, wt: f32, acc: &mut [f32]) {
        let id = self.map.as_ref().map_or(id, |m| m[id] as usize);
        if id >= self.rows { return; }
        let wt = wt * self.w.as_ref().map_or(1.0, |w| w[id]);
        let esz = if self.dt == Dtype::F32 { 4 } else { 2 };
        let b = &self.mm[self.off + id * self.dim * esz..][..self.dim * esz];
        for (a, c) in acc.iter_mut().zip(b.chunks_exact(esz)) {
            *a += wt * match self.dt {
                Dtype::F32 => f32::from_le_bytes(c.try_into().unwrap()),
                Dtype::F16 => half::f16::from_le_bytes(c.try_into().unwrap()).to_f32(),
                _ => half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32(),
            };
        }
    }

    /// L2-normalised mean embedding.
    pub fn embed(&self, s: &str) -> Vec<f32> {
        let mut acc = vec![0f32; self.dim];
        if let Ok(e) = self.tok.encode_fast(s, false) {
            let ids = e.get_ids();
            let ids = &ids[..ids.len().min(MAXTOK)];
            for &i in ids { self.add(i as usize, 1.0 / ids.len() as f32, &mut acc); }
        }
        let n = acc.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        acc.iter_mut().for_each(|x| *x /= n);
        acc
    }
}

/// Sign bits of (v - mean), packed to u64 words.
pub fn bits(v: &[f32], mean: &[f32]) -> Vec<u64> {
    v.chunks(64).zip(mean.chunks(64)).map(|(a, m)| a.iter().zip(m).enumerate().fold(0u64, |w, (i, (x, m))| w | (((x - m) > 0.0) as u64) << i)).collect()
}

pub fn cos(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
