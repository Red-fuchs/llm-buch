import json, subprocess, sys, time, re, os, urllib.request
SYS = open('src/main.rs').read().split('pub const SYS: &str = "')[1].split('";\n')[0].replace('\\"','"')
LF = "http://127.0.0.1:8102/v1/chat/completions"; GM = "http://127.0.0.1:8092/v1/chat/completions"
KEY = json.load(open(os.path.expanduser('~/.config/buch/config.json'))).get('key','')
QS = ["Wieso müssen freie Gesellschaften zwangsläufig Freiheitsfeinde ausschliessen und physisch entfernen?",
      "Was kritisiert Rothbard an Keynes?", "Welche Klassen unterscheidet Hoppe und wie entstehen sie?",
      "Was versteht man unter Agorismus?", "Warum sind Ökonomen in der Öffentlichkeit unbeliebt?"]
def chat(url, model, msgs, t=0.2, mx=900, key=None):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    b = json.dumps({"model": model, "temperature": t, "max_tokens": mx, "messages": msgs}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, b, h), timeout=300))
    return r["choices"][0]["message"]["content"].strip(), time.time() - t0
def passages(q, k=20):
    o = subprocess.run(["buch", "ask", q, "-k", str(k), "--json", "--no-llm"], capture_output=True, text=True, env=os.environ)
    return json.loads(o.stdout.strip().splitlines()[-1])["passages"]
def ctx(ps, n, ch, hdr=False):
    out = []
    for i, p in enumerate(ps[:n]):
        t = p["text"][:ch]
        out.append(f"[Quelle {i+1}] Autor: {p['author']} | Werk: {p['work']} | Zitierform: {p['citation']}\n{t}\n" if hdr else f"{p['citation']}\n{t}\n")
    return "\n".join(out)
STRICT = ("Du beantwortest Fragen nur anhand der nummerierten Quellen. Regeln: 1) Jeder Satz gibt genau eine Quelle wieder und endet mit deren Zitierform in eckigen Klammern, exakt wie angegeben. "
 "2) Der Autor eines Satzes ist immer der Autor der zitierten Quelle (siehe Zeile 'Autor'). 3) Erfinde keine Verbindungen, Gründe oder Folgerungen zwischen verschiedenen Quellen; verbinde Quellen nur durch Aufzählung. "
 "4) Wörtliche Zitate in „…“. 5) Steht die Antwort nicht in den Quellen, schreibe nur: Das steht nicht in der Bibliothek. 6) Deutsch, höchstens 6 Sätze, keine eigene Meinung.")
EXTR = ("Wähle aus den nummerierten Quellen die 2 bis 4 Sätze, die die Frage direkt beantworten. Gib für jeden: Quellennummer, dann den Satz wörtlich in „…“. Keine Erklärung. Beantwortet keine Quelle die Frage, schreibe nur: KEINE")
def run(v, q, ps):
    if v == "a": return chat(LF, "lfm3", [{"role":"system","content":SYS},{"role":"user","content":f"Passagen:\n\n{ctx(ps,10,2000)}\nFrage: {q}"}]), ctx(ps,10,2000)
    if v == "b": c = ctx(ps,20,1400); return chat(LF, "lfm3", [{"role":"system","content":SYS},{"role":"user","content":f"Passagen:\n\n{c}\nFrage: {q}"}]), c
    if v == "c": c = ctx(ps,5,2000); return chat(LF, "lfm3", [{"role":"system","content":SYS},{"role":"user","content":f"Passagen:\n\n{c}\nFrage: {q}"}]), c
    if v == "d": c = ctx(ps,6,1500,True); return chat(LF, "lfm3", [{"role":"system","content":STRICT},{"role":"user","content":f"Quellen:\n\n{c}\nFrage: {q}"}], 0.1), c
    if v == "d2":
        c = ctx(ps,8,1500,True); x, t1 = chat(LF, "lfm3", [{"role":"system","content":EXTR},{"role":"user","content":f"Quellen:\n\n{c}\nFrage: {q}"}], 0.0, 500)
        if x.strip().startswith("KEINE"): return ("Das steht nicht in der Bibliothek.", t1), c
        a, t2 = chat(LF, "lfm3", [{"role":"system","content":STRICT},{"role":"user","content":f"Quellen (nur diese Auszüge):\n\n{c}\nAusgewählte Auszüge:\n{x}\n\nFrage: {q}\nSchreibe die Antwort nur aus den ausgewählten Auszügen."}], 0.1)
        return (a, t1 + t2), c
    if v == "e": c = ctx(ps,10,1500,True); return chat(LF, "lfm3", [{"role":"system","content":STRICT},{"role":"user","content":f"Quellen:\n\n{c}\nFrage: {q}"}], 0.1), c
    if v == "g": c = ctx(ps,10,2000); return chat(GM, "gemma-4-26B-A4B-it-JANG_4M", [{"role":"system","content":SYS},{"role":"user","content":f"Passagen:\n\n{c}\nFrage: {q}"}], 0.2, 900, KEY), c
RUB = ("Du bist ein strenger Prüfer. Gegeben: Frage, die dem Modell vorgelegten Passagen, die Antwort. Bewerte jeweils 0, 1 oder 2:\n"
 "faith: jede Aussage der Antwort wird von der zitierten Passage wirklich getragen (0 = erfundene/unbelegte Behauptungen oder erfundene Verbindungen zwischen Quellen, 1 = kleinere Ungenauigkeiten, 2 = alles belegt)\n"
 "attr: Autor/Quelle korrekt zugeordnet (0 = falscher Autor oder falsche Quelle bei einer Aussage, 2 = alle richtig)\n"
 "abstain: wenn die Passagen die Frage nicht oder nur teilweise beantworten, sagt die Antwort das (2); behauptet sie trotzdem Vollständigkeit/Unbelegtes (0); wenn die Frage voll beantwortbar ist und die Antwort das tut, 2; verweigert fälschlich 0\n"
 "german: sprachliche Qualität (0 schlecht, 2 flüssig und korrekt)\n"
 'Antworte NUR mit JSON: {"faith":n,"attr":n,"abstain":n,"german":n,"note":"ein Satz Begründung mit dem schwersten Fehler"}')
def judge(q, c, a):
    x, _ = chat(GM, "gemma-4-26B-A4B-it-JANG_4M", [{"role":"system","content":RUB},{"role":"user","content":f"Frage: {q}\n\nPassagen:\n{c}\n\nAntwort:\n{a}"}], 0.0, 400, KEY)
    m = re.search(r"\{.*\}", x, re.S)
    try: return json.loads(m.group(0))
    except Exception: return {"faith":-1,"attr":-1,"abstain":-1,"german":-1,"note":x[:100]}
if __name__ == "__main__":
    vs = sys.argv[1].split(","); out = sys.argv[2]; res = []
    for q in QS:
        ps = passages(q)
        for v in vs:
            (a, t), c = run(v, q, ps)
            j = judge(q, c, a)
            res.append({"q": q, "v": v, "ans": a, "s": round(t, 1), "j": j}); print(v, q[:40], round(t,1), j, flush=True)
            json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
