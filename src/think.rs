//! Splits leaked plain-text reasoning from the answer: `<reasoning><channel|><answer>`.
const TOKS: [&str; 3] = ["<|channel>", "<channel|>", "<turn|>"];
const MARK: &str = "<channel|>";
pub const NO_ANSWER: &str = "Modell lieferte keine Antwort (nur Planungstext, abgebrochen).";
const LABELS: [&str; 6] = ["**Antwort**", "Antwort:", "Endgültige Antwort", "Ausformulierung:", "Ausführung:", "Final answer"];

/// no marker came: pull the final answer out of planning text (text after the last label), only if it ends like a finished sentence
pub fn salvage(s: &str) -> Option<String> {
    let (i, l) = LABELS.iter().filter_map(|l| s.rfind(l).map(|i| (i, *l))).max_by_key(|x| x.0)?;
    let a = s[i + l.len()..].trim_start_matches(|c: char| c == '*' || c == ':' || c.is_whitespace()).trim();
    (a.chars().count() > 20 && a.ends_with(['.', '!', '?', ')', ']'])).then(|| a.to_string())
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode { Auto, On, Off, /// On + planning-text salvage / error when no marker (heretic)
    Strict }

impl Mode {
    /// flag/env value wins, then config `strip_thinking` (string or bool), default auto
    pub fn resolve(flag: Option<&str>, cfg: &serde_json::Value) -> Mode {
        let c = &cfg["strip_thinking"];
        let v = flag.map(str::to_string).or_else(|| c.as_str().map(str::to_string)).or_else(|| c.as_bool().map(|b| b.to_string()));
        match v.as_deref().map(|x| x.trim().to_lowercase()).as_deref() {
            Some("on" | "true" | "1" | "yes") => Mode::On,
            Some("off" | "false" | "0" | "no") => Mode::Off,
            _ => Mode::Auto,
        }
    }
}

#[derive(PartialEq)]
enum St { Undecided, Live, Buffer }

pub struct Splitter { st: St, buf: String, acc: String, sent: usize, min_buf: usize, window: usize, reset: bool, strict: bool, pub thinking: String }

fn strip(s: &str) -> String { TOKS.iter().fold(s.to_string(), |a, t| a.replace(t, "")) }

/// length of the longest suffix of `s` that is a proper prefix of a token
fn held(s: &str) -> usize {
    TOKS.iter().flat_map(|t| (1..t.len()).rev().filter(|&n| s.ends_with(&t[..n])).take(1)).max().unwrap_or(0)
}

impl Splitter {
    /// `min_buf`: chars held back in auto mode before going live (0 = web); `window`: chars in which a marker still switches to buffering
    pub fn new(mode: Mode, min_buf: usize, window: usize) -> Self {
        let st = match mode { Mode::Auto => St::Undecided, Mode::On | Mode::Strict => St::Buffer, Mode::Off => St::Live };
        Splitter { st, buf: String::new(), acc: String::new(), sent: 0, min_buf, window, reset: false, strict: mode == Mode::Strict, thinking: String::new() }
    }
    pub fn is_thinking(&self) -> bool { self.st == St::Buffer }
    /// true once if already-emitted text must be discarded
    pub fn take_reset(&mut self) -> bool { std::mem::take(&mut self.reset) }
    /// separate reasoning_content delta
    pub fn reasoning(&mut self, t: &str) { self.thinking.push_str(t); }

    fn live(&mut self) -> String {
        let cut = self.buf.len() - held(&self.buf);
        let out = strip(&self.buf[..cut]);
        self.buf.drain(..cut);
        out
    }

    /// feed a chunk; returns answer text ready to show
    pub fn push(&mut self, c: &str) -> String {
        if self.st == St::Undecided {
            self.acc.push_str(c);
            if self.acc.contains("<|channel>") || self.acc.contains(MARK) {
                if self.sent > 0 { self.reset = true; }
                self.buf = std::mem::take(&mut self.acc);
                self.st = St::Buffer;
            } else if self.acc.len() >= self.min_buf {
                let cut = (self.acc.len() - held(&self.acc)).max(self.sent);
                let out = strip(&self.acc[self.sent..cut]);
                self.sent = cut;
                if self.acc.len() >= self.window {
                    self.buf = self.acc[cut..].to_string();
                    self.acc.clear();
                    self.st = St::Live;
                }
                return out;
            } else { return String::new(); }
        } else { self.buf.push_str(c); }
        if self.st == St::Buffer {
            let Some(i) = self.buf.rfind(MARK) else { return String::new() };
            self.thinking = strip(self.buf[..i].trim_start_matches("<|channel>thought").trim());
            self.buf = self.buf[i + MARK.len()..].trim_start().to_string();
            self.st = St::Live;
        }
        self.live()
    }

    /// end of stream; returns remaining answer text
    pub fn finish(&mut self) -> String {
        match self.st {
            St::Undecided => { let o = strip(&self.acc[self.sent..]); self.acc.clear(); self.st = St::Live; o }
            St::Live => strip(&std::mem::take(&mut self.buf)),
            St::Buffer => {
                let b = std::mem::take(&mut self.buf);
                let b = b.trim_start();
                let b = b.strip_prefix("<|channel>thought").or_else(|| b.strip_prefix("<|channel>")).unwrap_or(b);
                self.st = St::Live;
                let b = strip(b).trim().to_string();
                if self.strict { salvage(&b).unwrap_or_else(|| NO_ANSWER.into()) } else { b }
            }
        }
    }
}

/// non-streaming: (thinking, answer)
pub fn split_full(s: &str) -> (String, String) {
    let mut sp = Splitter::new(Mode::On, 0, 0);
    let mut a = sp.push(s);
    a.push_str(&sp.finish());
    (sp.thinking, a.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(chunks: &[&str]) -> (String, String) { run_m(Mode::On, chunks) }
    fn run_m(m: Mode, chunks: &[&str]) -> (String, String) {
        let mut sp = Splitter::new(m, 0, 0);
        let mut a: String = chunks.iter().map(|c| sp.push(c)).collect();
        a.push_str(&sp.finish());
        (sp.thinking.clone(), a)
    }
    #[test] fn salvage_label() { assert_eq!(salvage("Plan: x\nAntwort: Der Staat ist ein Zwangsmonopolist [1]."), Some("Der Staat ist ein Zwangsmonopolist [1].".into())); }
    #[test] fn salvage_truncated() { assert_eq!(salvage("Plan\nAntwort: Der Staat ist ein Zwangsmonopolist und die"), None); }
    #[test] fn salvage_none() { assert_eq!(salvage("Synthese der Passagen:\n1. foo"), None); }
    #[test] fn no_marker_gives_error() { assert_eq!(run_m(Mode::Strict, &["Plan:\n1. Passagen sichten\n2. Synthese, die gerade abbr"]).1, NO_ANSWER); }
    #[test] fn no_marker_label() { assert_eq!(run_m(Mode::Strict, &["Plan...\n**Antwort**\nStaat ist Monopol [1]. Ende."]).1, "Staat ist Monopol [1]. Ende."); }
    #[test] fn basic() { assert_eq!(run(&["2 + 2 = 4.<channel|>4"]), ("2 + 2 = 4.".into(), "4".into())); }
    #[test] fn split_marker() {
        assert_eq!(run(&["think ", "<chan", "nel|", ">Ant", "wort"]), ("think".into(), "Antwort".into()));
    }
    #[test] fn char_by_char() {
        let s = "<|channel>thought\nhmm<channel|>Hallo <turn|>";
        let c: Vec<String> = s.chars().map(String::from).collect();
        let c: Vec<&str> = c.iter().map(|x| x.as_str()).collect();
        assert_eq!(run(&c), ("hmm".into(), "Hallo ".into()));
    }
    #[test] fn no_marker() { assert_eq!(run(&["<|channel>thought\nnur ", "Antwort"]).1, "nur Antwort"); }
    #[test] fn no_marker_plain() { assert_eq!(run(&["Das ist ", "alles"]), ("".into(), "Das ist alles".into())); }
    #[test] fn multiple_markers_in_buffer() { assert_eq!(run(&["a<channel|>b<channel|>c"]).1, "c"); }
    #[test] fn stray_later() { assert_eq!(run(&["x<channel|>ab<|chan", "nel>cd<channel|>ef"]).1, "abcdef"); }
    #[test] fn lone_lt() { assert_eq!(run(&["x<channel|>a < b"]).1, "a < b"); }
    #[test] fn full() { assert_eq!(split_full("t<channel|> 4 "), ("t".into(), "4".into())); }
    #[test] fn auto_no_marker_live() {
        let mut sp = Splitter::new(Mode::Auto, 0, 2000);
        assert_eq!(sp.push("Hallo "), "Hallo ");
        assert_eq!(sp.push("Welt"), "Welt");
        assert!(!sp.take_reset());
        assert_eq!(sp.finish(), "");
    }
    #[test] fn auto_marker_early_resets() {
        let mut sp = Splitter::new(Mode::Auto, 0, 2000);
        assert_eq!(sp.push("denke "), "denke ");
        assert_eq!(sp.push("nach<chan"), "nach");
        assert!(!sp.take_reset());
        assert_eq!(sp.push("nel|>Ant"), "Ant");
        assert!(sp.take_reset());
        assert!(!sp.take_reset());
        assert_eq!(sp.push("wort"), "wort");
        assert_eq!(sp.thinking, "denke nach");
    }
    #[test] fn auto_late_marker_stripped() {
        let mut sp = Splitter::new(Mode::Auto, 0, 5);
        let mut o = sp.push("abcdefgh");
        o.push_str(&sp.push("<channel|>ij"));
        o.push_str(&sp.finish());
        assert_eq!(o, "abcdefghij");
        assert!(!sp.take_reset());
    }
    #[test] fn auto_cli_buffers_then_live() {
        let mut sp = Splitter::new(Mode::Auto, 10, 10);
        assert_eq!(sp.push("kurz"), "");
        assert_eq!(sp.push("er Text hier"), "kurzer Text hier");
        assert_eq!(sp.push(" mehr"), " mehr");
    }
    #[test] fn auto_short_finish() { let mut sp = Splitter::new(Mode::Auto, 200, 200); assert_eq!(sp.push("ja"), ""); assert_eq!(sp.finish(), "ja"); }
    #[test] fn off_live() { let mut sp = Splitter::new(Mode::Off, 0, 0); assert_eq!(sp.push("a<turn|>b"), "ab"); }
    #[test] fn reasoning_collected() { let mut sp = Splitter::new(Mode::Auto, 0, 100); sp.reasoning("x"); sp.reasoning("y"); assert_eq!(sp.thinking, "xy"); }
    #[test] fn mode_resolve() {
        use serde_json::json;
        assert_eq!(Mode::resolve(None, &json!({})), Mode::Auto);
        assert_eq!(Mode::resolve(None, &json!({"strip_thinking": true})), Mode::On);
        assert_eq!(Mode::resolve(Some("off"), &json!({"strip_thinking": "on"})), Mode::Off);
    }
}
