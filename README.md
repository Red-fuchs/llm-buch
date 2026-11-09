# llm-buch – Frag die libertäre Bibliothek (Prototyp)

> **PROTOTYP.** Experimentell, nicht für öffentliches Hosting gedacht. Antworten können falsch sein – Zitate und Seitenzahlen immer im Original prüfen.

`buch` ist eine lokale Open-RAG-Suche über die deutschen Übersetzungen libertärer Werke auf [lib-lib.org](https://lib-lib.org) (Übersetzungen größtenteils von Andreas Tank). Du stellst eine Frage, bekommst die passendsten Seiten mit Quelle (`[Hoppe: Die Privatisierung der Verteidigung, S. 34]`) und optional eine Antwort eines Sprachmodells, die sich nur auf diese Seiten stützen soll.

## Lizenz und Texte

- Der **Code** steht unter MIT (siehe `LICENSE`).
- Die **Buchtexte sind nicht Teil dieses Repos** und nicht von dieser Lizenz abgedeckt. `buch fetch` lädt die PDFs von lib-lib.org; es gelten deren jeweilige Lizenzen (teils Creative Commons, teils unbekannt). Index und PDFs bleiben lokal. Die Lizenz pro Werk zeigt `buch` bei jeder Fundstelle an.
- Der Suchkern (`src/kern/`) wurde aus einem privaten Projekt übernommen.

## Aufbau

- Quelle: PDFs -> seitenweise bereinigter Text (Swift/PDFKit-Helfer `tools/pdftext.swift`, nur macOS) -> ~2,6k Seiten, ~6,9k Passagen.
- Suche: BM25 (tantivy) + dichte Vektoren (multilingual-e5-small, MLX) per Reciprocal Rank Fusion. Optional Reranker (Liquid d1-2B) und Query-Expansion, beide standardmäßig aus.
- Antwort: beliebiger OpenAI-kompatibler Endpunkt (LM Studio, Ollama, llama.cpp, mlx_lm, oder Cloud wie OpenRouter). Experimentell: kleine lokale Modelle (`--answer lfm3|heretic`).
- Fällt ein Modell aus, zeigt `buch` trotzdem die Passagen.

## Schnellstart (macOS, Apple Silicon)

```sh
# Voraussetzungen: Rust (rustup), Xcode Command Line Tools (swiftc), python3 mit mlx
python3 -m venv ~/.local/share/buch/venv && ~/.local/share/buch/venv/bin/pip install mlx numpy tokenizers huggingface_hub && ~/.local/share/buch/venv/bin/hf download intfloat/multilingual-e5-small
# lokaler Modellserver mit OpenAI-API, z.B. LM Studio auf :1234 (oder Cloud-Key in den Einstellungen)
./scripts/install.sh            # baut Release, installiert nach ~/.local/bin
buch fetch && buch build && buch dense
buch web                        # http://127.0.0.1:8100/
```

Daten liegen in `~/AI-Model/buch` (`--dir` oder `BUCH_DIR`). Ohne `buch dense` fällt die Suche auf BM25 + Model2Vec zurück (`WIKI_MODEL`, Standard `~/AI-Model/models/minishlab/potion-multilingual-128M`, muss dann vorhanden sein).

## Befehle

| Befehl | Zweck |
|---|---|
| `buch fetch` | Katalog + PDFs laden (sequentiell, 2 MB/s) |
| `buch build` | PDF -> Seiten -> Index |
| `buch dense` | Passagen mit e5-small einbetten |
| `buch ask "Frage" [-k 10] [--no-llm] [--json] [--answer big\|lfm3\|heretic] [--rerank none\|d1]` | Fundstellen + Antwort |
| `buch serve` | JSON-Zeilen auf stdin/stdout für andere Programme |
| `buch web [--port 8100]` | lokale Oberfläche (nur 127.0.0.1) |
| `buch eval eval/buch.tsv` | Trefferquote pro erwartetem Werk |

## Modell-Einstellungen

Im Web-Dialog oder in `~/.config/buch/config.json` (Rechte 0600, Schlüssel wird nie zurückgegeben): `endpoint`, `model`, `key`, `answer`, `rerank`, `expand`. Umgebung: `BUCH_ENDPOINT`, `BUCH_KEY`, `BUCH_ANSWER`, `BUCH_RERANK`. Lokal bleibt alles auf deinem Rechner; bei Cloud-Anbietern gehen deine Fragen und die Fundstellen dorthin.

Optionale Hilfsserver (`scripts/buch-srv`, `buch-llm`, Ports 8101-8103) lesen zusätzliche Pfade/Befehle aus `~/.config/buch/env` (`LLAMA_SERVER`, `BUCH_MLX_LM`, `BUCH_PY`, `BUCH_PORTS`, `BUCH_GPU_WRAP`). Sie sind für den Normalbetrieb nicht nötig.

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
- Nur macOS/Apple Silicon getestet; keine Kapitelerkennung.

## Roadmap

- Zitat-Grammatik (llguidance): Modell darf nur wörtliche Zitate aus den Passagen mit gültiger Quelle ausgeben.
- Verlauf und Feedback in der Oberfläche.
- Mehr Bücher, bessere Titelseiten-/Verzeichnisfilterung.
