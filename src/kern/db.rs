use crate::kern::{emb::Emb, store::Store};
use anyhow::{Context, Result};
use memmap2::Mmap;
use std::{fs::File, path::{Path, PathBuf}};
use tantivy::{Index, tokenizer::*};

pub fn analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(StopWordFilter::new(Language::German).unwrap())
        .filter(AsciiFoldingFilter)
        .filter(Stemmer::new(Language::German))
        .build()
}

pub fn raw() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default()).filter(RemoveLongFilter::limit(40)).filter(LowerCaser).filter(AsciiFoldingFilter).build()
}

pub fn reg(i: &Index) { i.tokenizers().register("de", analyzer()); i.tokenizers().register("raw", raw()); }

pub fn mmap(p: &Path) -> Result<Mmap> {
    let f = File::open(p).with_context(|| p.display().to_string())?;
    // zero-length files cannot be mapped (BM25-only index has an empty vec.bin)
    if f.metadata()?.len() == 0 { return Ok(memmap2::MmapOptions::new().len(8).map_anon()?.make_read_only()?); }
    Ok(unsafe { Mmap::map(&f)? })
}

pub struct Db {
    pub dir: PathBuf,
    pub store: Store,
    pub idx: Index,
    pub arts: Mmap,
    pub titles: Mmap,
    pub tsort: Mmap,
    pub vecs: Mmap,
    pub mean: Vec<f32>,
    pub npass: usize,
    pub emb: Option<Emb>,
    pub alias: Option<Mmap>,
    pub sem: Option<crate::kern::sem::Sem>,
}

pub struct Art { pub id: usize, pub first: usize, pub n: usize, pub src: u32, pub title: String }

impl Db {
    pub fn open(dir: &Path, model: &Path) -> Result<Self> {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).context("no index; run `wiki build`")?)?;
        let idx = Index::open_in_dir(dir.join("tantivy"))?;
        reg(&idx);
        let emb = Emb::load(model).ok();
        Ok(Self {
            store: Store::open(dir)?, idx, arts: mmap(&dir.join("arts.dat"))?, titles: mmap(&dir.join("titles.txt"))?, tsort: mmap(&dir.join("tsort.idx"))?,
            vecs: mmap(&dir.join("vec.bin"))?, mean: serde_json::from_value(meta["mean"].clone())?, npass: meta["passages"].as_u64().unwrap() as usize, emb, alias: mmap(&dir.join("aliases.tsv")).ok(), sem: if std::env::var("WIKI_NOSEM").is_ok() { None } else { crate::kern::sem::Sem::open(dir) }, dir: dir.into(),
        })
    }

    pub fn alias(&self, span: &str) -> Option<Art> {
        let t = crate::kern::alias::lookup(self.alias.as_deref()?, &crate::kern::alias::norm(span))?;
        self.find(t)
    }

    pub fn narts(&self) -> usize { self.arts.len() / 16 - 1 }
    fn u(&self, i: usize, f: usize) -> u32 { u32::from_le_bytes(self.arts[i * 16 + f * 4..][..4].try_into().unwrap()) }

    pub fn art(&self, i: usize) -> Art {
        let (a, b) = (self.u(i, 2) as usize, self.u(i + 1, 2) as usize);
        Art { id: i, first: self.u(i, 0) as usize, n: self.u(i, 1) as usize, src: self.u(i, 3), title: String::from_utf8_lossy(&self.titles[a..b]).into() }
    }

    /// Article containing passage pid (articles hold contiguous passages).
    pub fn art_of(&self, pid: usize) -> Art {
        let (mut lo, mut hi) = (0, self.narts());
        while hi - lo > 1 { let m = (lo + hi) / 2; if self.u(m, 0) as usize <= pid { lo = m } else { hi = m } }
        self.art(lo)
    }

    pub fn find(&self, title: &str) -> Option<Art> {
        let t = title.trim().to_lowercase();
        let n = self.tsort.len() / 4;
        let at = |k: usize| self.art(u32::from_le_bytes(self.tsort[k * 4..][..4].try_into().unwrap()) as usize);
        let (mut lo, mut hi) = (0, n);
        while lo < hi { let m = (lo + hi) / 2; if at(m).title.to_lowercase() < t { lo = m + 1 } else { hi = m } }
        (lo < n).then(|| at(lo)).filter(|a| a.title.to_lowercase() == t)
    }
}

pub fn url(a: &Art) -> String {
    let h = if a.src == 1 { "de.wikivoyage.org" } else { "de.wikipedia.org" };
    format!("https://{h}/wiki/{}", a.title.replace(' ', "_"))
}
