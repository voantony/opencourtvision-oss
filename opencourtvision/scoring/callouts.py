"""Score tracking from spoken callouts (English + Spanish keywords).

Players call the score before serving. We transcribe the quiet window before each rally (mlx-whisper on
Apple Silicon), parse the callout, and follow the score point by point, flagging callouts that are not
reachable from the previous one. What the players called is never overwritten.
"""
import json
import os
import re
import subprocess

from .. import privacy
from ..detection.audio import read_wav
from ..ffmpeg import ffmpeg_bin
from ..utils import hms

DEFAULT_MODEL = os.environ.get("OCV_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

NUM = {"love": 0, "nada": 0, "zero": 0, "cero": 0, "fifteen": 15, "quince": 15, "thirty": 30, "treinta": 30,
       "forty": 40, "cuarenta": 40}
PIDX = {0: 0, 15: 1, 30: 2, 40: 3}
DIGITS = {"0", "00", "15", "30", "40"}


ORD = {"1": 1, "one": 1, "uno": 1, "una": 1, "first": 1, "primera": 1, "primer": 1, "2": 2, "two": 2, "dos": 2, "second": 2,
       "segunda": 2, "segundo": 2, "3": 3, "three": 3, "tres": 3, "third": 3, "tercera": 3, "tercer": 3}
_ORDW = "|".join(sorted(ORD, key=len, reverse=True))


def parse_callout(text):
    """Spoken score call -> one of
      ('deciding', 'golden' | 'star')       "punto de oro", "golden point" | "star point", "punto estrella", "punto de estrella"
      ('deuce_n', n)                        padel Star Point deuces: "deuce 1/2/3", "iguales dos", "deuce three"
      ('ad_n', (n, side or None))           "ventaja 1", "advantage two receiver", "ventaja dos resto"
      ('deuce', None) | ('ad', side)        unnumbered deuce / advantage ("deuce", "iguales", "cuarenta iguales", "ad in", "ventaja resto")
      ('pts', (server, receiver))           "thirty fifteen", "quince iguales" (= 15-15), "thirty all"
      None"""
    t = text.lower()
    if re.search(r"\b(star point|punto (de )?estrella)\b", t):
        return ("deciding", "star")
    if re.search(r"\b(punto de oro|golden point|punto oro)\b", t):
        return ("deciding", "golden")
    m = re.search(rf"\b(deuce|iguales|cuarenta iguales)\b\s*(?:n(?:o|º|umber|úmero)?\.?\s*)?({_ORDW})\b", t) or \
        re.search(rf"\b({_ORDW})\s+(deuce|iguales)\b", t)
    if m:
        n = ORD[m.group(2)] if m.group(2) in ORD else ORD[m.group(1)]
        return ("deuce_n", n)
    m = re.search(rf"\b(ad(?:vantage)?|ventaja)\b\s*({_ORDW})\b\s*(in|out|server|receiver|resto|saque)?", t)
    if m:
        return ("ad_n", (ORD[m.group(2)], m.group(3)))
    # Whisper usually writes tennis scores as digits ("30-15", "15 all", "15-0"): accept 0/15/30/40 as numerals too.
    ws = [NUM[w] if w in NUM else int(w) for w in re.findall(r"[a-záéíóúñ]+|\d+", t) if w in NUM or w in DIGITS]
    if len(ws) == 1 and ws[0] != 40 and re.search(r"\b(all|iguales)\b", t):
        return ("pts", (ws[0], ws[0]))                       # "quince iguales", "treinta iguales", "thirty all"
    if re.search(r"\b(deuce|iguales|cuarenta iguales)\b", t):
        return ("deuce", None)
    m = re.search(r"\b(ad(vantage)?|ventaja)\b\s*(in|out|server|receiver|resto|saque)?", t)
    if m:
        return ("ad", m.group(3))
    if len(ws) >= 2:
        return ("pts", (ws[0], ws[1]))
    if len(ws) == 1 and ws[0] == 40 and re.search(r"\ball\b", t):
        return ("deuce", None)
    return None


def describe_call(p):
    """Readable score_before for non-numeric calls: 'deuce 2', 'ad 1 resto', 'golden point', 'star point', 'ad in'."""
    kind, val = p
    if kind == "deciding": return f"{val} point"
    if kind == "deuce_n": return f"deuce {val}"
    if kind == "ad_n": return f"ad {val[0]}" + (f" {val[1]}" if val[1] else "")
    return kind + (f" {val}" if val else "")


def detect_deuce_rule(texts):
    """Padel deuce rule implied by spoken calls: 'golden_point' | 'star_point' | None (no evidence).
    Star evidence: "star point"/"punto estrella", a numbered deuce >= 2 or a numbered advantage. Golden: "punto de oro".
    If both appear, the majority wins; on a tie, the latest call."""
    votes = {"golden_point": 0, "star_point": 0}; last = None
    for x in texts:
        p = parse_callout(x if isinstance(x, str) else x.get("text", ""))
        if not p: continue
        k, v = p
        if k == "deciding": rule = "golden_point" if v == "golden" else "star_point"
        elif (k == "deuce_n" and v >= 2) or k == "ad_n": rule = "star_point"
        else: continue
        votes[rule] += 1; last = rule
    if not any(votes.values()):
        return None
    if votes["golden_point"] == votes["star_point"]:
        return last
    return max(votes, key=votes.get)


def stt_model(model=DEFAULT_MODEL):
    """Model mlx-whisper may load. Beta mode: a local directory only (bundled `whisper` or an explicit path), with
    HF_HUB_OFFLINE=1 set, else None (STT disabled). mlx-whisper treats anything else as a huggingface.co repo id."""
    if not privacy.beta_mode():
        return model
    os.environ.update(privacy.OFFLINE_ENV)
    for m in (model, privacy.bundled("whisper")):
        if m and os.path.isdir(m):
            return m
    return None


def transcribe_callouts(R, start, wav16k, workdir, model=DEFAULT_MODEL, run_stt=True):
    """Transcribe the window before each rally. Cached in workdir/stt_cache.json; with run_stt=False only the cache is read."""
    cache = os.path.join(workdir, "stt_cache.json")
    C = json.load(open(cache)) if os.path.exists(cache) else {}
    if not run_stt:
        return C
    model = stt_model(model)
    if model is None:
        print("beta mode: Whisper weights are not on the device, so spoken-score STT is off (no Hugging Face download)")
        return C
    import mlx_whisper  # Apple Silicon only
    nw = os.path.join(workdir, "audio_16k_norm.wav")
    if not os.path.exists(nw):
        subprocess.run([ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-y", "-i", wav16k, "-af",
                        "highpass=f=150,lowpass=f=5000,dynaudnorm=f=150:g=15:p=0.9", nw], check=True)
    sr, x = read_wav(nw)
    for i, r in enumerate(R):
        if r["t0"] < start - 30:
            continue
        a = max(r["t0"] - 12, R[i - 1]["t1"] + 0.8 if i else 0)
        b = r["t0"] - 0.15
        k = f"{a:.2f}"
        if b - a < 1 or k in C:
            continue
        o = mlx_whisper.transcribe(x[int(a * sr):int(b * sr)], path_or_hf_repo=model, word_timestamps=True,
                                   condition_on_previous_text=False, temperature=0.0)
        C[k] = dict(a=a, b=b, first_hit=r["t0"], lang=o.get("language"), text=o["text"].strip())
    json.dump(C, open(cache, "w"))
    return C


def score_timeline(segs, C, start):
    """Follow spoken callouts (server-first). A callout = score BEFORE the next point. Resync on every callout;
    flag a discrepancy when a callout is not reachable from the previous one given the points played in between."""
    calls = []
    for v in C.values():
        p = parse_callout(v["text"])
        if p and v["first_hit"] >= start - 1:
            calls.append((v["first_hit"], v["text"], p))
    rows, last = [], None
    for i, s in enumerate(segs):
        c = next((x for x in calls if s["A"] - 0.5 <= x[0] <= s["B"]), None)
        row = dict(point=i + 1, start=s["A"], callout="", score_before="?", conf="none", flag="")
        if c:
            _, txt, (kind, val) = c
            row["callout"] = txt.replace('"', "'")
            row["conf"] = "callout"
            if kind == "pts":
                row["score_before"] = f"{val[0]}-{val[1]} (server first)"
                cur = (PIDX.get(val[0]), PIDX.get(val[1]))
                if last and None not in cur and None not in last[1]:
                    played = i - last[0]
                    need = (cur[0] - last[1][0]) + (cur[1] - last[1][1])
                    if need < 0 or need > played:
                        row["flag"] = f"discrepancy: needs {need} points since point {last[0] + 1}, {played} played (new game or human scoring error?)"
                last = (i, cur)
            else:
                row["score_before"] = describe_call((kind, val))
                last = None
        rows.append(row)
    return rows


def write_score_csv(rows, path):
    with open(path, "w") as f:
        f.write("point,start_s,start_hms,score_before,confidence,callout_heard,flag\n")
        for r in rows:
            f.write(f"{r['point']},{r['start']:.3f},{hms(r['start'])},{r['score_before']},{r['conf']},\"{r['callout']}\",{r['flag']}\n")


def found_callouts(C, start):
    out = []
    for _, v in sorted(C.items(), key=lambda kv: kv[1]["a"]):
        p = parse_callout(v["text"])
        if p and v["first_hit"] >= start - 30:
            out.append((round(v["first_hit"], 1), v["text"], p))
    return out
