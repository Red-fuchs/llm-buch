"""buch reranker: Liquid d1-2B decision model, pointwise P(relevant) per (query, passage).
POST /rank {"query": str, "passages": [str], "mode": "noul"|"score"} -> {"scores": [float], "ms": int}; GET /health.
Usage: python buch-rerank-srv.py PORT   (run through buch-rerank)"""
import sys, os, json, time, pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
if os.environ.get("BUCH_MLXVLM_PATH"): sys.path.insert(0, os.environ["BUCH_MLXVLM_PATH"])  # optional mlx-vlm checkout with d1 support
from mlx_vlm import load

D1 = pathlib.Path(os.environ.get("BUCH_D1", str(pathlib.Path.home() / "AI-Model/models/LiquidAI/d1-2B-MLX-8bit")))
Q = {
    "noul": {"r": {"type": "noul", "instructions": "Enthält die Textstelle Aussagen, die die Frage des Nutzers direkt beantworten oder belegen? Die Wortwahl der Frage und der Textstelle kann verschieden sein."}},
    "score": {"r": {"type": "score", "instructions": "Wie gut beantwortet die Textstelle die Frage des Nutzers? Die Wortwahl kann verschieden sein.", "criteria": ["irrelevant", "gleiches Thema, aber keine Antwort", "teilweise Antwort", "direkte Antwort"]}},
}
m, _ = load(str(D1))
lock = Lock()


def rank(q, ps, mode, clip=800, bs=8):
    rq = [(f"Frage: {q}\n\nTextstelle: {p[:clip]}", Q[mode]) for p in ps]
    out = []
    for i in range(0, len(rq), bs):
        for r in m.system_one_batch(rq[i:i + bs]):
            a = r["answers"]["r"]
            out.append(float(a["noul"] if mode == "noul" else a["score"]))
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, f, *a):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), f % a)); sys.stderr.flush()

    def _j(self, c, o):
        b = json.dumps(o).encode(); self.send_response(c); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._j(200, {"ok": True})

    def do_POST(self):
        try:
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            t = time.perf_counter()
            with lock:
                s = rank(b["query"], b["passages"], b.get("mode", "noul"), int(b.get("clip", 800)), int(b.get("bs", 8)))
            self._j(200, {"scores": s, "ms": round((time.perf_counter() - t) * 1000)})
        except Exception as e:
            self._j(500, {"error": str(e)})


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
