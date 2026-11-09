#!/usr/bin/env python3
"""Generic multilingual-e5 (BERT/XLM-R, MLX fp16) for buch.
  dense.py embed MODEL_DIR passages.jsonl OUT_DIR   -> OUT_DIR/dense.f16 (n x D, row = line), dense.json {dim,model,n}
  dense.py serve MODEL_DIR SOCK                      -> unix socket; client sends one query line, gets D f32 LE
Memory capped (MLX cache/limit) after the 2026-11-03 jetsam incident."""
import json, os, sys, time, socket, fcntl, numpy as np, mlx.core as mx, mlx.nn as nn
from tokenizers import Tokenizer
if os.environ.get('DENSE_CPU'): mx.set_default_device(mx.cpu)
mx.set_cache_limit(512 * 2**20); mx.set_memory_limit(int(float(os.environ.get("DENSE_GB", 4)) * 2**30))

class Layer(nn.Module):
    def __init__(s, D, H, FF, eps):
        super().__init__(); s.H, s.D = H, D
        s.q, s.k, s.v, s.o = (nn.Linear(D, D) for _ in range(4))
        s.ln1, s.ln2 = nn.LayerNorm(D, eps=eps), nn.LayerNorm(D, eps=eps)
        s.i, s.out = nn.Linear(D, FF), nn.Linear(FF, D)
    def __call__(s, x, mask):
        B, T, D = x.shape; H = s.H
        sp = lambda t: t.reshape(B, T, H, D // H).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(sp(s.q(x)), sp(s.k(x)), sp(s.v(x)), scale=(D // H) ** -0.5, mask=mask)
        x = s.ln1(x + s.o(a.transpose(0, 2, 1, 3).reshape(B, T, D)))
        return s.ln2(x + s.out(nn.gelu(s.i(x))))

class M(nn.Module):
    def __init__(s, c):
        super().__init__(); D = c["hidden_size"]; eps = c["layer_norm_eps"]
        s.off = 2 if c["model_type"] == "xlm-roberta" else 0
        s.we, s.pe, s.te = nn.Embedding(c["vocab_size"], D), nn.Embedding(c["max_position_embeddings"], D), nn.Embedding(c["type_vocab_size"], D)
        s.eln = nn.LayerNorm(D, eps=eps)
        s.layers = [Layer(D, c["num_attention_heads"], c["intermediate_size"], eps) for _ in range(c["num_hidden_layers"])]
    def __call__(s, ids, am):
        x = s.eln(s.we(ids) + s.pe(mx.arange(ids.shape[1]) + s.off)[None] + s.te(mx.zeros_like(ids)))
        mask = mx.where(am[:, None, None, :] > 0, 0.0, -1e4).astype(x.dtype)
        for l in s.layers: x = l(x, mask)
        m = am[..., None].astype(x.dtype)
        p = ((x * m).sum(1) / m.sum(1)).astype(mx.float32)
        return p / mx.linalg.norm(p, axis=-1, keepdims=True)

class Embedder:
    def __init__(s, d, maxlen=512):
        c = json.load(open(f"{d}/config.json")); s.D = c["hidden_size"]; s.maxlen = maxlen
        w = mx.load(f"{d}/model.safetensors"); w = {k.removeprefix("bert.").removeprefix("roberta."): v for k, v in w.items()}
        nw = {"we.weight": w["embeddings.word_embeddings.weight"], "pe.weight": w["embeddings.position_embeddings.weight"],
              "te.weight": w["embeddings.token_type_embeddings.weight"], "eln.weight": w["embeddings.LayerNorm.weight"], "eln.bias": w["embeddings.LayerNorm.bias"]}
        for i in range(c["num_hidden_layers"]):
            p = f"encoder.layer.{i}."
            for a, b in [("attention.self.query", "q"), ("attention.self.key", "k"), ("attention.self.value", "v"), ("attention.output.dense", "o"), ("intermediate.dense", "i"), ("output.dense", "out"),
                         ("attention.output.LayerNorm", "ln1"), ("output.LayerNorm", "ln2")]:
                for t in ("weight", "bias"): nw[f"layers.{i}.{b}.{t}"] = w[f"{p}{a}.{t}"]
        s.m = M(c); s.m.load_weights(list(nw.items()), strict=True); s.m.apply(lambda a: a.astype(mx.float16)); s.m.eval()
        s.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
    def embed(s, texts, prefix, bs=32):
        enc = s.tok.encode_batch([prefix + t for t in texts])
        ids = [e.ids[:s.maxlen - 1] + [2] if len(e.ids) > s.maxlen else e.ids for e in enc]
        order = np.argsort([len(i) for i in ids]); out = np.zeros((len(ids), s.D), np.float32)
        for st in range(0, len(ids), bs):
            ix = order[st:st + bs]; T = max(len(ids[i]) for i in ix)
            a = np.ones((len(ix), T), np.int32); am = np.zeros((len(ix), T), np.int32)
            for r, i in enumerate(ix): a[r, :len(ids[i])] = ids[i]; am[r, :len(ids[i])] = 1
            out[ix] = np.array(s.m(mx.array(a), mx.array(am)))
        return out

if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "embed":
        rows = [json.loads(l) for l in open(sys.argv[3])]; od = sys.argv[4]; os.makedirs(od, exist_ok=True)
        em = Embedder(d); t = time.time()
        v = em.embed([f"{r['title'].split(': ',1)[0]}. {r['text']}" for r in rows], "passage: ")
        v.astype("<f2").tofile(f"{od}/dense.f16")
        json.dump({"path": os.path.abspath(d), "dim": em.D, "model": os.path.basename(d.rstrip('/')), "n": len(rows)}, open(f"{od}/dense.json", "w"))
        print(f"{len(rows)} x {em.D} in {time.time()-t:.0f}s")
    else:
        sock = sys.argv[3]; IDLE = float(os.environ.get("EMBD_IDLE", 600))
        lock = open(sock + ".lock", "w")
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError: time.sleep(20); sys.exit(0)
        em = Embedder(d, 64); em.embed(["warm"], "query: ")
        if os.path.exists(sock): os.unlink(sock)
        s = socket.socket(socket.AF_UNIX); s.bind(sock); s.listen(8); s.settimeout(5); last = time.time()
        try:
            while time.time() - last < IDLE:
                try: c, _ = s.accept()
                except socket.timeout: continue
                last = time.time()
                with c:
                    c.settimeout(5); buf = b""
                    try:
                        while not buf.endswith(b"\n") and len(buf) < 4096:
                            x = c.recv(4096)
                            if not x: break
                            buf += x
                        c.sendall(em.embed([buf.decode("utf8", "replace").strip()], "query: ")[0].astype("<f4").tobytes())
                    except Exception as e: print("err", e, file=sys.stderr)
        finally:
            try: os.unlink(sock)
            except OSError: pass
