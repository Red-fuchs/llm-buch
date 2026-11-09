# llm-buch – Frag die libertäre Bibliothek (Prototyp)

> **PROTOTYP.** Experimentell, nicht für öffentliches Hosting gedacht. Antworten können falsch sein – Zitate und Seitenzahlen immer im Original prüfen.

`buch` ist eine lokale Open-RAG-Suche über die deutschen Übersetzungen libertärer Werke auf [lib-lib.org](https://lib-lib.org) (Übersetzungen größtenteils von Andreas Tank). Du stellst eine Frage, bekommst die passendsten Seiten mit Quelle (`[Hoppe: Die Privatisierung der Verteidigung, S. 34]`) und optional eine Antwort eines Sprachmodells, die sich nur auf diese Seiten stützen soll.

## Lizenz und Texte

- Der **Code** steht unter MIT (siehe `LICENSE`).
- Die **Buchtexte sind nicht Teil dieses Repos** und nicht von dieser Lizenz abgedeckt. `buch fetch` lädt die PDFs von lib-lib.org; es gelten deren jeweilige Lizenzen (teils Creative Commons, teils unbekannt). Index und PDFs bleiben lokal. Die Lizenz pro Werk zeigt `buch` bei jeder Fundstelle an.
- Der Suchkern (`src/kern/`) wurde aus einem privaten Projekt übernommen.

## Aufbau

- Quelle: PDFs -> seitenweise bereinigter Text (reines Rust, `pdf-extract`; optional `pdftotext` aus poppler oder PDFKit-Helfer auf macOS) -> ~2,6k Seiten, ~6,9k Passagen.
- Suche: BM25 (tantivy) + dichte Vektoren (multilingual-e5-small, läuft in-process mit candle auf der CPU, kein Python) per Reciprocal Rank Fusion. Optional Reranker (Liquid d1-2B) und Query-Expansion, beide standardmäßig aus.
- Antwort: beliebiger OpenAI-kompatibler Endpunkt (LM Studio, Ollama, llama.cpp, mlx_lm, oder Cloud wie OpenRouter). Experimentell: kleine lokale Modelle (`--answer lfm3|heretic`).
- Fällt ein Modell aus, zeigt `buch` trotzdem die Passagen.

## Plattformen

| | macOS (Apple Silicon) | Linux (x86_64) | Windows (x86_64) |
|---|---|---|---|
| `buch fetch/build/dense/ask/web/serve/eval` | ja | ja | ja |
| PDF-Text | Rust (Standard), optional PDFKit/`pdftotext` | Rust (Standard), optional `pdftotext` | Rust (Standard), optional `pdftotext` |
| Embeddings | candle (CPU, Accelerate) | candle (CPU) | candle (CPU) |
| Antwortmodell (OpenAI-kompatibel) | LM Studio, Ollama, llama.cpp, Cloud | dito | dito |
| Optionale Hilfsserver `scripts/buch-srv`, `buch-llm`, `buch-rerank-srv.py` (MLX / llama.cpp) | ja (Extras) | nur mit zsh + llama.cpp, ungetestet | nicht unterstützt |
| Voraussetzungen | Rust (rustup) | Rust (rustup), C-Compiler (`build-essential`), `curl` | Rust (rustup, MSVC Build Tools), `curl.exe` (in Windows 10/11 enthalten) |

Ausgeliefert wird eine einzelne Binärdatei. Python, Xcode oder swiftc sind für den Standardweg **nicht** nötig. Getestet: macOS (Apple Silicon) lokal; Linux im Container; Windows über GitHub Actions (CI baut und testet auf allen drei).

PDF-Extraktor wählen: `BUCH_PDF=rust` (Standard; fällt bei einem Fehler auf `pdftotext` zurück, falls im PATH), `pdftotext` (poppler) oder `pdfkit` (nur macOS; Helfer aus `tools/pdftext.swift` mit `swiftc -O tools/pdftext.swift -o ~/.local/bin/buch-pdftext` bauen). Auf drei Testbüchern stimmt die Rust-Extraktion mit PDFKit zu über 99,8 % der Wörter pro Seite überein, gleiche Seitenzahlen.

Beim ersten `buch dense` (bzw. der ersten Suche mit Vektoren) lädt `buch` das Modell `intfloat/multilingual-e5-small` von Hugging Face (ca. 470 MB, einmalig, in den HF-Cache; `HF_HOME` / `HF_HUB_CACHE` werden beachtet; liegt es dort schon, bleibt alles offline; `HF_HUB_OFFLINE=1` verbietet den Download; `BUCH_DL_LIMIT=2.5` begrenzt auf 2,5 MB/s; `BUCH_E5_DIR` zeigt auf ein fertiges Modellverzeichnis). Das Einbetten der ~6,9k Passagen dauert auf einer CPU einige Minuten.

## Schnellstart

Zuerst Rust installieren (https://rustup.rs) und ein Antwortmodell bereitstellen: lokal z.B. [LM Studio](https://lmstudio.ai) (Server auf :1234) oder [Ollama](https://ollama.com) (`http://localhost:11434/v1`), oder Cloud über einen [OpenRouter](https://openrouter.ai)-Schlüssel (`https://openrouter.ai/api/v1`). Endpunkt, Modell und Schlüssel trägst du im Web-Dialog ein. Ohne Modell zeigt `buch ask --no-llm` nur die Fundstellen.

**macOS / Linux**

```sh
git clone https://github.com/Red-fuchs/llm-buch && cd llm-buch
./scripts/install.sh            # baut Release, installiert nach ~/.local/bin (BUCH_EXTRAS=1: auch die optionalen Hilfsserver)
buch fetch && buch build && buch dense
buch web                        # http://127.0.0.1:8100/
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Red-fuchs/llm-buch; cd llm-buch
.\scripts\install.ps1           # cargo build --release, kopiert buch.exe nach %LOCALAPPDATA%\Programs\buch und trägt den Ordner in den Benutzer-PATH ein
# neues Terminal öffnen (PATH), dann:
buch fetch; buch build; buch dense
buch web                        # http://127.0.0.1:8100/
```

Fertige Binärdateien für Linux, Windows und macOS gibt es (sobald veröffentlicht) unter GitHub Releases.

Daten: `--dir` oder `BUCH_DIR`; Standard ist `~/AI-Model/buch`, falls vorhanden (Altbestand), sonst das Datenverzeichnis der Plattform (Linux `~/.local/share/buch`, macOS `~/Library/Application Support/buch`, Windows `%APPDATA%\buch`). Ohne `buch dense` läuft die Suche nur mit BM25 (oder zusätzlich Model2Vec über `WIKI_MODEL`, falls vorhanden; veraltet).

## Befehle

| Befehl | Zweck |
|---|---|
| `buch fetch` | Katalog + PDFs laden (sequentiell, 2 MB/s) |
| `buch build` | PDF -> Seiten -> Index |
| `buch dense [--out DIR]` | Passagen mit e5-small einbetten (candle, CPU; `BUCH_GPU_WRAP` = optionaler Queue-Wrapper) |
| `buch ask "Frage" [-k 10] [--no-llm] [--json] [--answer big\|lfm3\|heretic] [--rerank none\|d1]` | Fundstellen + Antwort |
| `buch serve` | JSON-Zeilen auf stdin/stdout für andere Programme |
| `buch web [--port 8100]` | lokale Oberfläche (nur 127.0.0.1) |
| `buch eval eval/buch.tsv` | Trefferquote pro erwartetem Werk |

## Modell-Einstellungen

Im Web-Dialog oder in `config.json` (`~/.config/buch/` falls vorhanden, sonst Konfigurationsordner der Plattform, z.B. `%APPDATA%\buch\`; unter Unix Rechte 0600, Schlüssel wird nie zurückgegeben): `endpoint`, `model`, `key`, `answer`, `rerank`, `expand`. Umgebung: `BUCH_ENDPOINT`, `BUCH_KEY`, `BUCH_ANSWER`, `BUCH_RERANK`. Lokal bleibt alles auf deinem Rechner; bei Cloud-Anbietern gehen deine Fragen und die Fundstellen dorthin.

Optionale **macOS-Extras** (`scripts/buch-srv`, `buch-llm`, `buch-rerank-srv.py`, Ports 8101-8103; MLX / llama.cpp, zsh; Installation mit `BUCH_EXTRAS=1 ./scripts/install.sh`; unter Windows nicht unterstützt) lesen zusätzliche Pfade/Befehle aus `~/.config/buch/env` (`LLAMA_SERVER`, `BUCH_MLX_LM`, `BUCH_PY`, `BUCH_PORTS`, `BUCH_GPU_WRAP`). Sie sind für den Normalbetrieb nicht nötig.

## Messwerte (19 Fragen, `eval/buch.tsv`, Treffer: richtiges Werk unter den Top-N)

| Suche | @1 | @3 | @6 | @10 |
|---|---|---|---|---|
| Model2Vec (alt) | 11 | 13 | 14 | 17 |
| **BM25 + e5-small (Standard)** | 15 | 17 | 17 | 17 |
| nur e5-small | 14 | 15 | 18 | 18 |
| nur e5-base | 13 | 16 | 17 | 19 |
| BM25 + e5-small + d1-Rerank (30) | 14 | 17 | 17 | 17 |

19 Fragen: eine Frage = 5 Punkte, Unterschiede von 1-2 sind Rauschen.

## Bekannte Einschränkungen

- Kleine Modelle (LFM3-1.7B, Gemma-4-E4B) ordnen Zitate gern falschen Autoren zu und lassen Quellenangaben weg. Größere Modelle (z.B. Gemma-4 26B) sind deutlich zuverlässiger, aber nicht fehlerfrei.
- Titelseiten und Verzeichnisse erzeugen Rauschen; die angegebene Seite ist die PDF-Seite, die gedruckte kann um 1 abweichen.
- Manche Fragen verfehlen das richtige Werk (z.B. Mises-Fragen).
- Die Embeddings laufen nur auf der CPU (kein GPU-Pfad); die Hilfsserver sind nur für macOS gedacht. Keine Kapitelerkennung.

## Roadmap

- Zitat-Grammatik (llguidance): Modell darf nur wörtliche Zitate aus den Passagen mit gültiger Quelle ausgeben.
- Verlauf und Feedback in der Oberfläche.
- Mehr Bücher, bessere Titelseiten-/Verzeichnisfilterung.
