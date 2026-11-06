"""Hard gates for the Vamos MVP edit (Media Director, mvp_edit_style.md §9.2), configured per sport.

Input: an edit-eval JSON (examples/edit_eval.example.json) per labelled match:
  {"sport": "padel",
   "labels":  {"points": [{"id", "serve_contact", "motion_start", "t_dec", "t_dec_tier"}], "warmup": [[a, b], ...]},
   "outputs": {"full_cut": [{"start", "end"}], "reel_beats": [{"start", "end", "point_id"}],
               "other_clips": [{"start", "end", "output"}]},
   "metrics": {"captions_total", "captions_wrong", "caption_precision_wide", "bug_wrong_scores",
               "jerk_p95_px", "jerk_baseline_px" (or "reel_stats": the reel's stats JSON), "reel_dead_time_max_s",
               "full_cut_gap_dead_s", "slowmo_max", "punch_in_max", "captions_per_8s_max", "best_point_today_s",
               "consent_violations"}}
Clip-interval gates are computed from labels + outputs; the rest are read from metrics. A gate whose input is missing
is reported "missing" and the overall result is INCOMPLETE (all gates must pass).
"""
import json

# Per-sport parameters (§8 per-sport table). Thresholds are the same for every sport in §9.2.
SPORT_PARAMS = {
    "tennis": dict(pre_roll_s=4.0, in_play_gap_s=3.0),
    "padel": dict(pre_roll_s=3.0, in_play_gap_s=4.0),
    "pickleball": dict(pre_roll_s=3.0, in_play_gap_s=3.0),
}
COMMON = dict(start_tol_s=0.5, start_ok_frac=0.98, cutoff_frac=0.02, cutoff_buffer_s=dict(reel_beat=0.4, point=1.0, highlight=1.0), missed_frac=0.01, spurious_frac=0.02,
              caption_precision_wide=0.95, jerk_ratio=0.5, reel_dead_s=1.0, full_cut_gap_s=0.0, slowmo_max=2, punch_in_max=3,
              captions_per_8s=1, best_point_today_s=600.0)


def sport_config(sport, **overrides):
    sport = sport.lower()
    if sport not in SPORT_PARAMS:
        raise ValueError(f"unknown sport {sport!r}; choose tennis, padel or pickleball")
    c = dict(COMMON, **SPORT_PARAMS[sport], sport=sport); c.update(overrides)
    return c


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def edit_metrics(labels, outputs, cfg):
    """Clip-interval metrics from labelled points and the produced clips."""
    pts = labels.get("points", []); full = outputs.get("full_cut", []); beats = outputs.get("reel_beats", [])
    m = {}
    if pts and full:
        errs, clipped = [], 0
        for p in pts:
            c = max(full, key=lambda k: _overlap(k["start"], k["end"], p["serve_contact"], p["t_dec"]))
            if _overlap(c["start"], c["end"], p["serve_contact"], p["t_dec"]) <= 0:
                continue
            errs.append(abs(c["start"] - (p["serve_contact"] - cfg["pre_roll_s"])))
            if c["start"] > p.get("motion_start", p["serve_contact"]) + 1e-9:
                clipped += 1
        m["start_within_tol_frac"] = sum(e <= cfg["start_tol_s"] + 1e-9 for e in errs) / len(errs) if errs else 0.0
        m["clipped_serves"] = clipped
        missed = [p["id"] for p in pts if not any(_overlap(c["start"], c["end"], p["serve_contact"], p["t_dec"]) > 0 for c in full)]
        m["missed_frac"] = len(missed) / len(pts); m["missed_ids"] = missed
        m["spurious_frac"] = sum(not any(_overlap(c["start"], c["end"], p["serve_contact"], p["t_dec"]) > 0 for p in pts) for c in full) / len(full)
        # cut-off: a clip ends before the visible outcome (t_dec) + its clip type's buffer (MD decision 6:
        # reel_beat 0.4 s; point and highlight clips 1.0 s). Full-cut clips are point clips.
        buf = cfg["cutoff_buffer_s"]; cut, n = 0, 0
        for c in full:
            cov = [p for p in pts if _overlap(c["start"], c["end"], p["serve_contact"], p["t_dec"]) > 0]
            if cov:
                n += 1; cut += c["end"] < max(p["t_dec"] for p in cov) + buf["point"] - 1e-9
        byid = {p["id"]: p for p in pts}; reel_t1_cut = 0
        for b in beats:
            p = byid.get(b.get("point_id"))
            if p is None: continue
            n += 1; bad = b["end"] < p["t_dec"] + buf["reel_beat"] - 1e-9; cut += bad
            reel_t1_cut += bad and p.get("t_dec_tier") == "T1"
        for c in outputs.get("other_clips", []):       # top 3, highlights, best point today: whole-point clips
            p = byid.get(c.get("point_id"))
            if p is None: continue
            n += 1; cut += c["end"] < p["t_dec"] + buf.get(c.get("type", "highlight"), buf["highlight"]) - 1e-9
        m["cutoff_frac"] = cut / n if n else 0.0; m["reel_t1_cutoffs"] = reel_t1_cut
    warm = labels.get("warmup", [])
    if "warmup" in labels:
        clips = full + beats + outputs.get("other_clips", [])
        m["warmup_leak_s"] = round(sum(_overlap(c["start"], c["end"], a, b) for c in clips for a, b in warm), 3)
    return m


def _jerk(metrics):
    j, base = metrics.get("jerk_p95_px"), metrics.get("jerk_baseline_px")
    rs = metrics.get("reel_stats")
    if j is None and rs and rs.get("jerk_p95"):
        j = rs["jerk_p95"].get("px_per_frame3")          # float path (the rounded value is jerk_p95_rounded, debug only)
    if base is None and rs and rs.get("jerk_qa"):
        base = rs["jerk_qa"].get("baseline")
    return j, base


def check_gates(sport, metrics, cfg=None):
    """metrics: edit_metrics(...) merged with measured metrics. -> dict(sport, gates [..], overall PASS/FAIL/INCOMPLETE)."""
    c = cfg or sport_config(sport); M = metrics; G = []

    def gate(gid, name, value, ok, target, fmt="{}"):
        G.append(dict(id=gid, name=name, value=value, target=target,
                      status="missing" if value is None else ("pass" if ok else "fail"), shown=None if value is None else fmt.format(value)))

    v = M.get("start_within_tol_frac")
    gate("point_start", "Point-start accuracy (|start - (serve - pre)| <= 0.5 s)", v, v is not None and v >= c["start_ok_frac"],
         f">= {c['start_ok_frac']:.0%} of points (pre-roll {c['pre_roll_s']} s)", "{:.1%}")
    v = M.get("clipped_serves"); gate("clipped_serves", "Clipped serves", v, v == 0, "0")
    v = M.get("cutoff_frac"); gate("cutoffs", "Cut-off endings", v, v is not None and v <= c["cutoff_frac"] + 1e-12, f"<= {c['cutoff_frac']:.0%}", "{:.1%}")
    v = M.get("reel_t1_cutoffs"); gate("reel_t1_cutoffs", "Cut-offs on reel beats with T1 t_dec", v, v == 0, "0")
    v = M.get("missed_frac"); gate("missed_points", "Missed points", v, v is not None and v <= c["missed_frac"] + 1e-12, f"<= {c['missed_frac']:.0%}", "{:.1%}")
    v = M.get("spurious_frac"); gate("spurious_points", "Spurious points", v, v is not None and v <= c["spurious_frac"] + 1e-12, f"<= {c['spurious_frac']:.0%}", "{:.1%}")
    v = M.get("warmup_leak_s"); gate("warmup", "Warmup leakage (s in any output)", v, v is not None and v <= 0, "0", "{:.2f} s")
    v = M.get("captions_wrong"); gate("wrong_captions", "Wrong captions (eval set)", v, v == 0, "0")
    v = M.get("caption_precision_wide")
    gate("caption_precision", "Caption precision (wider set)", v, v is not None and v >= c["caption_precision_wide"], f">= {c['caption_precision_wide']}", "{:.3f}")
    v = M.get("bug_wrong_scores"); gate("bug_score", "Wrong score on the bug", v, v == 0, "0")
    j, base = _jerk(M)
    ratio = None if j is None or not base else j / base
    gate("jerk", "Camera jerk_p95 (float path) / v3 baseline", ratio, ratio is not None and ratio <= c["jerk_ratio"] + 1e-12,
         f"<= {c['jerk_ratio']} x v3", "{:.3f} x" + (f" ({j:.4g} vs {base:.4g} px/frame^3)" if ratio is not None else ""))
    v = M.get("reel_dead_time_max_s"); gate("reel_dead_time", "Longest dead run in reels", v, v is not None and v <= c["reel_dead_s"] + 1e-9, f"<= {c['reel_dead_s']} s", "{:.2f} s")
    v = M.get("full_cut_gap_dead_s"); gate("full_cut_gaps", "Dead footage between points (full cut)", v, v is not None and v <= c["full_cut_gap_s"] + 1e-9, "0 s", "{:.2f} s")
    v = M.get("slowmo_max"); gate("slowmo", "Slow-mo per reel (max)", v, v is not None and v <= c["slowmo_max"], f"<= {c['slowmo_max']}")
    v = M.get("punch_in_max"); gate("punch_ins", "Verified punch-ins per reel (max)", v, v is not None and v <= c["punch_in_max"], f"<= {c['punch_in_max']}")
    v = M.get("captions_per_8s_max"); gate("caption_rate", "Captions per 8 s (max)", v, v is not None and v <= c["captions_per_8s"], f"<= {c['captions_per_8s']}")
    v = M.get("best_point_today_s")
    gate("best_point_today", "Time to best point today (1-h match, M2 Pro)", v, v is not None and v <= c["best_point_today_s"], f"<= {c['best_point_today_s'] / 60:.0f} min", "{:.0f} s")
    v = M.get("consent_violations"); gate("consent", "Consent violations", v, v == 0, "0")
    st = [g["status"] for g in G]
    overall = "FAIL" if "fail" in st else "INCOMPLETE" if "missing" in st else "PASS"
    return dict(sport=c["sport"], overall=overall, gates=G, n_pass=st.count("pass"), n_fail=st.count("fail"), n_missing=st.count("missing"))


def evaluate_edit(doc, sport=None):
    """doc: one edit-eval dict (or a list of them, one per match; per-match metrics are pooled by worst case / mean)."""
    docs = doc if isinstance(doc, list) else [doc]
    sport = (sport or docs[0].get("sport", "tennis")).lower()
    cfg = sport_config(sport)
    per = []
    for d in docs:
        m = edit_metrics(d.get("labels", {}), d.get("outputs", {}), cfg); m.update(d.get("metrics", {})); per.append(m)
    if len(per) == 1:
        merged = per[0]
    else:   # worst case for counts/maxima, point-weighted for fractions
        merged = {}
        for k in {k for m in per for k in m}:
            vals = [m[k] for m in per if m.get(k) is not None]
            if not vals or k == "reel_stats": continue
            if isinstance(vals[0], list): merged[k] = [x for v in vals for x in v]
            elif k in ("start_within_tol_frac", "caption_precision_wide"): merged[k] = min(vals)
            elif k.endswith("_frac"): merged[k] = max(vals)
            else: merged[k] = max(vals)
    rep = check_gates(sport, merged, cfg); rep["matches"] = len(docs)
    return rep


def load_edit_eval(path):
    return json.load(open(path))


def format_gates(rep):
    L = [f"hard gates · {rep['sport']} · {rep.get('matches', 1)} match(es): {rep['overall']}  "
         f"(pass {rep['n_pass']}, fail {rep['n_fail']}, missing {rep['n_missing']})"]
    for g in rep["gates"]:
        mark = {"pass": "PASS", "fail": "FAIL", "missing": "----"}[g["status"]]
        L.append(f"  {mark}  {g['name']:<52} {g['shown'] if g['shown'] is not None else 'not measured':<34} target {g['target']}")
    return "\n".join(L)


def synthetic_edit_eval(sport="padel", seed=0):
    """A labelled toy match + outputs that pass every gate, for docs, tests and `ocv eval --synthetic-gates`."""
    import random
    rnd = random.Random(seed); pre = SPORT_PARAMS[sport]["pre_roll_s"]
    pts, full, beats, t = [], [], [], 600.0
    for i in range(120):
        sc = t + rnd.uniform(0, 3); dec = sc + rnd.uniform(4, 18)
        pts.append(dict(id=f"p{i + 1}", serve_contact=round(sc, 3), motion_start=round(sc - 1.2, 3), t_dec=round(dec, 3), t_dec_tier="T1"))
        full.append(dict(start=round(sc - pre + rnd.uniform(-0.3, 0.3), 3), end=round(dec + 1.6, 3)))
        t = dec + rnd.uniform(15, 30)
    beats = [dict(start=p["serve_contact"] - 1.0, end=p["t_dec"] + 0.8, point_id=p["id"]) for p in pts[::15]]
    return dict(sport=sport, labels=dict(points=pts, warmup=[[0.0, 560.0]]), outputs=dict(full_cut=full, reel_beats=beats),
                metrics=dict(captions_total=9, captions_wrong=0, caption_precision_wide=0.97, bug_wrong_scores=0, jerk_p95_px=0.71,
                             jerk_baseline_px=2.0, reel_dead_time_max_s=0.6, full_cut_gap_dead_s=0.0, slowmo_max=1, punch_in_max=2,
                             captions_per_8s_max=1, best_point_today_s=412, consent_violations=0))
