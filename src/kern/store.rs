use anyhow::Result;
use memmap2::Mmap;
use rayon::prelude::*;
use std::{fs::File, io::Write, path::Path};
use zstd::{bulk, dict::DecoderDictionary};

pub const BLK: usize = 64;
const LVL: i32 = 15;
const TRAIN: usize = 30_000;

/// Passages in blocks of BLK, zstd + trained dict. blocks.dat = [u32 rawlen][zstd], blocks.off = u64 per block (+end).
pub struct StoreW {
    dir: std::path::PathBuf,
    out: File,
    off: Vec<u64>,
    pos: u64,
    dict: Option<Vec<u8>>,
    cur: Vec<u8>,
    n: usize,
    raw: Vec<Vec<u8>>,
    samples: Vec<Vec<u8>>,
    pub rawbytes: u64,
}

impl StoreW {
    pub fn new(dir: &Path) -> Result<Self> {
        Ok(Self { dir: dir.into(), out: File::create(dir.join("blocks.dat"))?, off: vec![], pos: 0, dict: None, cur: vec![], n: 0, raw: vec![], samples: vec![], rawbytes: 0 })
    }

    pub fn push(&mut self, p: &[u8]) -> Result<()> {
        if self.dict.is_none() && self.samples.len() < TRAIN { self.samples.push(p.to_vec()); }
        self.rawbytes += p.len() as u64;
        if self.n > 0 { self.cur.push(0); }
        self.cur.extend_from_slice(p);
        self.n += 1;
        if self.n == BLK {
            self.raw.push(std::mem::take(&mut self.cur));
            self.n = 0;
            if self.dict.is_none() && self.samples.len() >= TRAIN { self.train()?; }
            if self.dict.is_some() && self.raw.len() >= 512 { self.flush()?; }
        }
        Ok(())
    }

    fn train(&mut self) -> Result<()> {
        let d = zstd::dict::from_samples(&self.samples, 112_640).unwrap_or_default();
        std::fs::write(self.dir.join("dict.zst"), &d)?;
        self.dict = Some(d);
        self.samples.clear();
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let d = self.dict.as_deref().unwrap_or(&[]);
        let blocks = std::mem::take(&mut self.raw);
        let comp: Vec<Vec<u8>> = blocks.par_iter().map_init(|| bulk::Compressor::with_dictionary(LVL, d).unwrap(), |c, b| {
            let mut v = (b.len() as u32).to_le_bytes().to_vec();
            v.extend(c.compress(b).unwrap());
            v
        }).collect();
        for v in comp { self.off.push(self.pos); self.out.write_all(&v)?; self.pos += v.len() as u64; }
        Ok(())
    }

    pub fn finish(mut self) -> Result<usize> {
        eprintln!("store raw {:.1} MB", self.rawbytes as f64 / 1e6);
        if self.n > 0 { self.raw.push(std::mem::take(&mut self.cur)); }
        if self.dict.is_none() { self.train()?; }
        self.flush()?;
        self.off.push(self.pos);
        std::fs::write(self.dir.join("blocks.off"), self.off.iter().flat_map(|o| o.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(self.off.len() - 1)
    }
}

pub struct Store { dat: Mmap, off: Mmap, dict: DecoderDictionary<'static> }

impl Store {
    pub fn open(dir: &Path) -> Result<Self> {
        let m = |n: &str| -> Result<Mmap> { Ok(unsafe { Mmap::map(&File::open(dir.join(n))?)? }) };
        Ok(Self { dat: m("blocks.dat")?, off: m("blocks.off")?, dict: DecoderDictionary::copy(&std::fs::read(dir.join("dict.zst"))?) })
    }

    fn o(&self, i: usize) -> usize { u64::from_le_bytes(self.off[i * 8..i * 8 + 8].try_into().unwrap()) as usize }

    /// Returns (section, text).
    pub fn get(&self, pid: usize) -> Option<(String, String)> {
        let b = pid / BLK;
        if (b + 1) * 8 >= self.off.len() { return None; }
        let blk = &self.dat[self.o(b)..self.o(b + 1)];
        let rl = u32::from_le_bytes(blk[..4].try_into().unwrap()) as usize;
        let raw = bulk::Decompressor::with_prepared_dictionary(&self.dict).ok()?.decompress(&blk[4..], rl).ok()?;
        let p = raw.split(|&c| c == 0).nth(pid % BLK)?;
        let s = String::from_utf8_lossy(p);
        let (sec, t) = s.split_once('\n')?;
        Some((sec.into(), t.into()))
    }
}
