//! Cross-platform locations: data dir, config file, home-relative paths, optional helper scripts.
use std::path::PathBuf;

pub fn home() -> PathBuf { dirs::home_dir().unwrap_or_default() }

/// expands a leading `~` (also `~/..`) to the home dir
pub fn ex(p: &str) -> PathBuf {
    match p.strip_prefix('~') {
        Some("") => home(),
        Some(r) if r.starts_with('/') || r.starts_with('\\') => home().join(&r[1..]),
        _ => PathBuf::from(p),
    }
}

/// data dir: BUCH_DIR / --dir, else existing `~/AI-Model/buch` (author's setup), else `<data_dir>/buch`
pub fn data_dir() -> PathBuf {
    let old = home().join("AI-Model").join("buch");
    if old.exists() { old } else { dirs::data_dir().unwrap_or_else(|| home().join(".local/share")).join("buch") }
}

/// config file: legacy `~/.config/buch/config.json` when present (macOS/Linux), else `<config_dir>/buch/config.json`
pub fn config_path() -> PathBuf {
    let old = home().join(".config").join("buch").join("config.json");
    if old.exists() { return old; }
    dirs::config_dir().unwrap_or_else(|| home().join(".config")).join("buch").join("config.json")
}

/// optional helper server script (buch-chat, buch-llm, ...): only on unix and only if installed in ~/.local/bin
pub fn helper_script(name: &str) -> Option<PathBuf> {
    if cfg!(not(unix)) { return None; }
    let p = home().join(".local/bin").join(name);
    p.exists().then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn tilde() {
        assert_eq!(ex("~/x/y"), home().join("x/y"));
        assert_eq!(ex("~"), home());
        assert_eq!(ex("/a/~b"), PathBuf::from("/a/~b"));
        assert_eq!(ex("~user/x"), PathBuf::from("~user/x"));
    }
}
