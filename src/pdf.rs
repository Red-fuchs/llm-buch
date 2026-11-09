//! PDF -> per-page text. Default: pure Rust (pdf-extract). BUCH_PDF=pdftotext (poppler) or =pdfkit (macOS helper) override.
use anyhow::{bail, Context, Result};
use std::{path::Path, process::Command};

pub type Pages = Vec<(u64, String)>;

pub fn pages(pdf: &Path) -> Result<Pages> {
    match std::env::var("BUCH_PDF").unwrap_or_default().as_str() {
        "pdftotext" => poppler(pdf),
        "pdfkit" => pdfkit(pdf),
        _ => rust(pdf).or_else(|e| poppler(pdf).map_err(|_| e)),
    }
}

fn rust(pdf: &Path) -> Result<Pages> {
    let p = pdf.to_path_buf();
    let h = std::thread::Builder::new().stack_size(64 << 20).spawn(move || pdf_extract::extract_text_by_pages(&p))?;
    let v = h.join().map_err(|_| anyhow::anyhow!("pdf-extract panicked"))??;
    Ok(v.into_iter().enumerate().map(|(i, t)| (i as u64 + 1, t)).collect())
}

/// poppler `pdftotext` (on PATH); pages are form-feed separated
fn poppler(pdf: &Path) -> Result<Pages> {
    let o = Command::new("pdftotext").args(["-enc", "UTF-8"]).arg(pdf).arg("-").output().context("pdftotext not on PATH")?;
    if !o.status.success() { bail!("pdftotext failed"); }
    let s = String::from_utf8_lossy(&o.stdout);
    let mut v: Vec<&str> = s.split('\x0c').collect();
    if v.last().is_some_and(|l| l.trim().is_empty()) { v.pop(); }
    Ok(v.into_iter().enumerate().map(|(i, t)| (i as u64 + 1, t.to_string())).collect())
}

#[cfg(target_os = "macos")]
fn pdfkit(pdf: &Path) -> Result<Pages> {
    let h = std::env::var("BUCH_PDFTEXT").map(|s| crate::paths::ex(&s)).unwrap_or_else(|_| crate::paths::home().join(".local/bin/buch-pdftext"));
    let o = Command::new(&h).arg(pdf).output().with_context(|| format!("PDFKit helper {} missing (compile tools/pdftext.swift)", h.display()))?;
    if !o.status.success() { bail!("pdftext failed"); }
    String::from_utf8_lossy(&o.stdout).lines().map(|l| { let v: serde_json::Value = serde_json::from_str(l)?; Ok((v["page"].as_u64().unwrap_or(0), v["text"].as_str().unwrap_or("").to_string())) }).collect()
}

#[cfg(not(target_os = "macos"))]
fn pdfkit(_: &Path) -> Result<Pages> { bail!("BUCH_PDF=pdfkit is macOS only") }
