use crate::kern::{chunk::chunk, db::reg, emb::{self, Emb}, store::StoreW};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::{fs::File, io::{BufRead, BufReader, Write}, path::{Path, PathBuf}};
use tantivy::{doc, schema::*, Index};

pub struct In { pub title: String, pub url: String, pub text: String }

fn row(v: &serde_json::Value) -> Option<In> {
    let g = |k: &str| v[k].as_str().map(String::from);
    Some(In { title: g("title")?, url: g("url").unwrap_or_default(), text: g("text")? })
}

fn json(p: &Path, f: &mut dyn FnMut(Vec<In>) -> Result<()>) -> Result<()> {
    if p.extension().is_some_and(|e| e == "jsonl") {
        let mut v = vec![];
        for l in BufReader::new(File::open(p)?).lines() {
            if let Some(r) = row(&serde_json::from_str(&l?)?) { v.push(r); }
            if v.len() >= 2000 { f(std::mem::take(&mut v))?; }
        }
        f(v)
    } else { // datasets-server page: {"rows":[{"row":{..}}]}
        let j: serde_json::Value = serde_json::from_reader(File::open(p)?)?;
        f(j["rows"].as_array().map(|r| r.iter().filter_map(|x| row(&x["row"])).collect()).unwrap_or_default())
    }
}

pub fn run(inputs: &[PathBuf], out: &Path, model: &Path, no_vec: bool) -> Result<()> {
    let _ = std::fs::remove_dir_all(out.join("tantivy"));
    std::fs::create_dir_all(out.join("tantivy"))?;
    let emb = if no_vec { None } else { Some(Emb::load(model)?) };
    let mut sb = SchemaBuilder::new();
    let ti = |tk: &str| TextOptions::default().set_indexing_options(TextFieldIndexing::default().set_tokenizer(tk).set_index_option(IndexRecordOption::WithFreqs));
    let f_title = sb.add_text_field("title", ti("de"));
    let f_tr = sb.add_text_field("tr", ti("raw"));
    let f_body = sb.add_text_field("body", ti("de"));
    let f_pid = sb.add_u64_field("pid", FAST);
    let idx = Index::create_in_dir(out.join("tantivy"), sb.build())?;
    reg(&idx);
    let mut iw = idx.writer(1_000_000_000)?;
    let mut store = StoreW::new(out)?;
    let mut vecf = File::create(out.join("vec.bin"))?;
    let (mut arts, mut titles, mut tl) = (File::create(out.join("arts.dat"))?, File::create(out.join("titles.txt"))?, Vec::<String>::new());
    let (mut pid, mut toff) = (0u32, 0u32);
    let mut mean: Option<Vec<f32>> = None;
    let t0 = std::time::Instant::now();

    let mut batch = |items: Vec<In>| -> Result<()> {
        type P = (String, String, Vec<f32>);
        let done: Vec<(In, Vec<P>)> = items.into_par_iter().filter(|a| !(a.text.len() < 300 && a.text.contains("REDIRECT"))).map(|a| {
            let ps = chunk(&a.text).into_iter().map(|p| {
                let v = emb.as_ref().map(|e| e.embed(&format!("{}. {}. {}", a.title, p.sec, p.text))).unwrap_or_default();
                (p.sec, p.text, v)
            }).collect();
            (a, ps)
        }).collect();
        if mean.is_none() {
            let vs: Vec<&Vec<f32>> = done.iter().flat_map(|d| d.1.iter().map(|p| &p.2)).filter(|v| !v.is_empty()).collect();
            let d = vs.first().map_or(0, |v| v.len());
            let mut m = vec![0f32; d];
            vs.iter().for_each(|v| m.iter_mut().zip(*v).for_each(|(a, b)| *a += b / vs.len() as f32));
            mean = Some(m);
        }
        let mean = mean.as_ref().unwrap();
        for (a, ps) in done {
            if ps.is_empty() { continue; }
            let hosts = if a.url.contains("wikivoyage") { 1u32 } else { 0 };
            for f in [pid, ps.len() as u32, toff, hosts] { arts.write_all(&f.to_le_bytes())?; }
            titles.write_all(a.title.as_bytes())?;
            toff += a.title.len() as u32;
            tl.push(a.title.to_lowercase());
            for (sec, text, v) in ps {
                store.push(format!("{sec}\n{text}").as_bytes())?;
                if !v.is_empty() { for w in emb::bits(&v, mean) { vecf.write_all(&w.to_le_bytes())?; } }
                iw.add_document(doc!(f_title => a.title.as_str(), f_tr => a.title.as_str(), f_body => format!("{sec} {text}"), f_pid => pid as u64))?;
                pid += 1;
            }
        }
        eprint!("\r{pid} passages, {:.0}s   ", t0.elapsed().as_secs_f32());
        Ok(())
    };

    for p in inputs {
        match p.extension().and_then(|e| e.to_str()) {
            Some("json" | "jsonl") => json(p, &mut batch)?,
            Some("zim") => bail!("zim import not built in (see README)"),
            _ => bail!("unsupported input {}", p.display()),
        }
    }
    eprintln!("\ncommitting tantivy...");
    iw.commit()?;
    let nb = store.finish()?;
    arts.write_all(&[pid, 0, toff, 0].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    let mut ord: Vec<u32> = (0..tl.len() as u32).collect();
    ord.sort_by(|a, b| tl[*a as usize].cmp(&tl[*b as usize]));
    std::fs::write(out.join("tsort.idx"), ord.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    std::fs::write(out.join("meta.json"), serde_json::to_vec(&serde_json::json!({"passages": pid, "articles": tl.len(), "blocks": nb, "mean": mean.unwrap_or_default(), "dim": emb.as_ref().map_or(0, |e| e.dim)}))?)?;
    eprintln!("done: {pid} passages, {} articles, {:.0}s", tl.len(), t0.elapsed().as_secs_f32());
    Ok(())
}
