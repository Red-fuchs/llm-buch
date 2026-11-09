//! Alias table: fakt (Wikidata) alias -> de article title. Sorted TSV "norm<TAB>title", binary-searched via mmap.
use anyhow::Result;
use std::{collections::HashMap, path::Path};

pub fn norm(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars().flat_map(|c| c.to_lowercase()) {
        match c {
            'ä' => o += "ae", 'ö' => o += "oe", 'ü' => o += "ue", 'ß' => o += "ss",
            'é' | 'è' | 'ê' | 'ë' | 'á' | 'à' | 'â' | 'ã' | 'å' | 'í' | 'ì' | 'î' | 'ï' | 'ó' | 'ò' | 'ô' | 'õ' | 'ú' | 'ù' | 'û' | 'ñ' | 'ç' | 'š' | 'č' | 'ž' | 'ł' | 'ø' | 'ý' | 'ć' | 'ń' | 'ś' | 'ź' | 'ż' | 'ă' | 'ş' | 'ț' | 'ğ' | 'ı' | 'ő' | 'ű' | 'ř' | 'ě' | 'ů' => o.push(match c {
                'é' | 'è' | 'ê' | 'ë' | 'ě' => 'e', 'á' | 'à' | 'â' | 'ã' | 'å' | 'ă' => 'a', 'í' | 'ì' | 'î' | 'ï' | 'ı' => 'i',
                'ó' | 'ò' | 'ô' | 'õ' | 'ø' | 'ő' => 'o', 'ú' | 'ù' | 'û' | 'ű' | 'ů' => 'u', 'ñ' | 'ń' => 'n', 'ç' | 'č' | 'ć' => 'c',
                'š' | 'ś' | 'ş' => 's', 'ž' | 'ź' | 'ż' => 'z', 'ł' => 'l', 'ý' => 'y', 'ř' => 'r', 'ț' => 't', _ => 'g' }),
            c if c.is_alphanumeric() => o.push(c),
            _ => if !o.ends_with(' ') && !o.is_empty() { o.push(' ') }
        }
    }
    o.trim_end().to_string()
}


pub fn lookup<'a>(tab: &'a [u8], key: &str) -> Option<&'a str> {
    let end = |s: usize| tab[s..].iter().position(|&b| b == b'\n').map_or(tab.len(), |x| s + x);
    let (mut lo, mut hi) = (0usize, tab.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let s = tab[..mid].iter().rposition(|&b| b == b'\n').map_or(0, |x| x + 1).max(lo);
        let e = end(s);
        let k = tab[s..e].split(|&b| b == b'\t').next().unwrap_or(&[]);
        if k < key.as_bytes() { lo = e + 1 } else { hi = s }
    }
    if lo >= tab.len() { return None; }
    let l = std::str::from_utf8(&tab[lo..end(lo)]).ok()?;
    let (k, t) = l.split_once('\t')?;
    (k == key).then_some(t)
}

pub fn build(dir: &Path, fakt: &Path) -> Result<()> {
    let titles = std::fs::read(dir.join("titles.txt"))?;
    let arts = std::fs::read(dir.join("arts.dat"))?;
    let n = arts.len() / 16 - 1;
    let u = |i: usize| u32::from_le_bytes(arts[i * 16 + 8..][..4].try_into().unwrap()) as usize;
    let mut set: HashMap<String, String> = HashMap::with_capacity(n);
    for i in 0..n { let t = String::from_utf8_lossy(&titles[u(i)..u(i + 1)]).into_owned(); set.insert(t.to_lowercase(), t); }
    let con = rusqlite::Connection::open_with_flags(fakt, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let mut st = con.prepare("SELECT n.norm, i.label, i.sl FROM nm n JOIN items i USING(q)")?;
    let mut best: HashMap<String, (i64, Option<String>)> = HashMap::new();
    let mut rows = st.query([])?;
    while let Some(r) = rows.next()? {
        let (al, lab, sl): (String, String, i64) = (r.get(0)?, r.get(1)?, r.get(2)?);
        if al.chars().count() < 3 || al == norm(&lab) || al.chars().all(|c| c.is_ascii_digit() || c == ' ') { continue; }
        if best.get(&al).is_some_and(|b| sl <= b.0) { continue; }
        // Wikidata "Friedrich II. von Preußen" vs dewiki "Friedrich II. (Preußen)"
        let t = set.get(&lab.to_lowercase()).or_else(|| lab.split_once(" von ").and_then(|(a, b)| set.get(&format!("{a} ({b})").to_lowercase())));
        best.insert(al, (sl, t.cloned()));
    }
    let mut v: Vec<_> = best.into_iter().filter_map(|(a, (s, t))| t.map(|t| (a, (s, t)))).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    let out: String = v.iter().map(|(a, (_, t))| format!("{a}\t{t}\n")).collect();
    std::fs::write(dir.join("aliases.tsv"), out)?;
    println!("{} aliases", v.len());
    Ok(())
}
