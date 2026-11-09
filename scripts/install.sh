#!/bin/zsh
set -e
cd "${0:A:h}/.."
B=~/.local/bin; S=~/.local/share/buch
mkdir -p $B $S
cargo build --release
rm -f $B/buch && cp target/release/buch $B/buch && codesign -s - -f $B/buch
cp tools/pdftext.swift embd/dense.py $S/
swiftc -O tools/pdftext.swift -o $B/buch-pdftext.new && rm -f $B/buch-pdftext && mv $B/buch-pdftext.new $B/buch-pdftext
for f in buch-srv buch-rerank-srv.py buch-llm; do rm -f $B/$f; cp scripts/$f $B/$f; chmod +x $B/$f; done
for l in buch-chat buch-rerank; do ln -sf buch-srv $B/$l; done
echo installed
