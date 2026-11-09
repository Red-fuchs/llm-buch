pub struct Pass { pub sec: String, pub text: String }

const MIN: usize = 100;
const MAX: usize = 200;

fn wc(s: &str) -> usize { s.split_whitespace().count() }

/// drop footnote markers like [12]
fn nofn(l: &str) -> String {
    let mut o = String::with_capacity(l.len());
    let mut it = l.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if c == '[' {
            if let Some(j) = l[i + 1..].find(']') { if j > 0 && j <= 4 && l[i + 1..i + 1 + j].bytes().all(|b| b.is_ascii_digit()) {
                while it.peek().is_some_and(|(k, _)| *k <= i + 1 + j) { it.next(); }
                continue;
            } }
        }
        o.push(c);
    }
    o
}

fn clean(l: &str) -> String {
    let l = &nofn(l);
    if l.starts_with('|') {
        let c: Vec<&str> = l.split('|').map(str::trim).filter(|c| !c.is_empty() && !c.chars().all(|ch| ch == '-' || ch == ':')).collect();
        c.join("; ")
    } else { l.trim().to_string() }
}

/// Split a long line at sentence ends into <=MAX word pieces.
fn split_long(l: &str, out: &mut Vec<String>) {
    let mut cur = String::new();
    for w in l.split_inclusive(' ') {
        cur.push_str(w);
        if wc(&cur) >= MIN && (w.ends_with(". ") || w.ends_with("? ") || w.ends_with("! ")) || wc(&cur) >= MAX {
            out.push(std::mem::take(&mut cur).trim().to_string());
        }
    }
    if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
}

/// finewiki markdown -> passages with section heading. Returns None for redirects/empty.
pub fn chunk(md: &str) -> Vec<Pass> {
    let mut res = vec![];
    let mut sec = String::new();
    let mut cur = String::new();
    let mut flush = |sec: &str, cur: &mut String, res: &mut Vec<Pass>| {
        if !cur.trim().is_empty() { res.push(Pass { sec: sec.to_string(), text: std::mem::take(cur).trim().to_string() }); }
        cur.clear();
    };
    for (i, l) in md.lines().enumerate() {
        if let Some(h) = l.strip_prefix('#') {
            if i == 0 { continue; }
            flush(&sec, &mut cur, &mut res);
            sec = h.trim_start_matches('#').trim().to_string();
            continue;
        }
        let l = clean(l);
        if l.is_empty() { continue; }
        let mut parts = vec![];
        if wc(&l) > MAX { split_long(&l, &mut parts) } else { parts.push(l) }
        for p in parts {
            if !cur.is_empty() && wc(&cur) + wc(&p) > MAX { flush(&sec, &mut cur, &mut res); }
            if !cur.is_empty() { cur.push('\n'); }
            cur.push_str(&p);
            if wc(&cur) >= MIN { flush(&sec, &mut cur, &mut res); }
        }
        // short trailing paragraph stays in cur and merges with next paragraph of same section
    }
    flush(&sec, &mut cur, &mut res);
    res
}
