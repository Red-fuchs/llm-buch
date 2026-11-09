#!/bin/sh
# macOS / Linux: build release and install `buch` to ~/.local/bin. Optional helper servers with BUCH_EXTRAS=1.
set -e
cd "$(dirname "$0")/.."
B="$HOME/.local/bin"
mkdir -p "$B"
cargo build --release
rm -f "$B/buch" && cp target/release/buch "$B/buch"
if [ "$(uname -s)" = Darwin ]; then codesign -s - -f "$B/buch" 2>/dev/null || true; fi
if [ -n "$BUCH_EXTRAS" ]; then
  for f in buch-srv buch-rerank-srv.py buch-llm; do rm -f "$B/$f"; cp "scripts/$f" "$B/$f"; chmod +x "$B/$f"; done
  for l in buch-chat buch-rerank; do ln -sf buch-srv "$B/$l"; done
  echo "optional helper servers installed (need zsh, llama.cpp / mlx_lm; see README)"
fi
echo "installed $B/buch"
case ":$PATH:" in *":$B:"*) ;; *) echo "note: add $B to your PATH" ;; esac
