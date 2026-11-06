"""Blind pairwise test vs SwingVision's auto-edit (Media Director, mvp_edit_style.md §9.1).

CSV, one row per rated pair: rater_id, output, match_id, sport, vote[, reason, killer_tag]
  output: reel | full_cut | top3   vote: ours | sv | tie  (also win/loss, 1/0/0.5). A tie counts as half a win.
Pass rule per output: win rate >= target AND comparisons >= minimum AND independent raters >= 30 AND the lower bound of the
two-sided 90% Wilson interval > 50% (z = 1.645; the doc's check "n = 30 at 70% -> ~0.55" uses this z).
Minimum comparisons: reel 30, full cut 80, top 3 40 (MD decision 5: 65% at n = 30 can't clear the Wilson rule).
"""
import csv
import math
from collections import defaultdict

Z90 = 1.6448536269514722   # two-sided 90% (one-sided 95%)
TARGETS = {
    "reel": dict(win_rate=0.70, min_comparisons=30, min_raters=30, label="Story reel (9:16)"),
    "full_cut": dict(win_rate=0.60, min_comparisons=80, min_raters=30, label="Full-match cut (16:9)"),
    "top3": dict(win_rate=0.65, min_comparisons=40, min_raters=30, label="Top 3 points"),
}
ALIASES = {"story_reel": "reel", "reel_9x16": "reel", "full": "full_cut", "full_match": "full_cut", "top_3": "top3"}
VOTE = {"ours": 1.0, "win": 1.0, "1": 1.0, "vamos": 1.0, "sv": 0.0, "swingvision": 0.0, "loss": 0.0, "0": 0.0, "tie": 0.5, "0.5": 0.5}


def wilson_lower(wins, n, z=Z90):
    """Lower Wilson score bound for wins/n (wins may be fractional for ties). n = 0 -> 0.0."""
    if n <= 0:
        return 0.0
    p = min(1.0, max(0.0, wins / n)); z2 = z * z
    centre = p + z2 / (2 * n)
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z2 / (4 * n * n)))
    return max(0.0, (centre - half) / (1 + z2 / n))


def wilson_upper(wins, n, z=Z90):
    if n <= 0:
        return 1.0
    return 1.0 - wilson_lower(n - wins, n, z)


def read_votes(path):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            out = r.get("output", "").lower(); out = ALIASES.get(out, out)
            v = r.get("vote", "").lower()
            if out not in TARGETS:
                raise ValueError(f"row {i + 2}: unknown output {r.get('output')!r} (reel, full_cut, top3)")
            if v not in VOTE:
                raise ValueError(f"row {i + 2}: unknown vote {r.get('vote')!r} (ours, sv, tie)")
            rows.append(dict(rater=r.get("rater_id", ""), output=out, match=r.get("match_id", ""), sport=r.get("sport", "").lower(),
                             score=VOTE[v], killer=r.get("killer_tag", "")))
    return rows


def _score(rows, t):
    n = len(rows); wins = sum(r["score"] for r in rows)
    raters = {r["rater"] for r in rows}
    rate = wins / n if n else 0.0; lo = wilson_lower(wins, n)
    checks = dict(win_rate=rate >= t["win_rate"] - 1e-12, comparisons=n >= t["min_comparisons"], raters=len(raters) >= t["min_raters"],
                  wilson=lo > 0.5)
    return dict(n=n, wins=wins, ties=sum(r["score"] == 0.5 for r in rows), raters=len(raters), win_rate=rate, wilson_lower_90=lo,
                wilson_upper_90=wilson_upper(wins, n), checks=checks, passed=all(checks.values()))


def score_blind(rows, exclude_raters=()):
    """-> dict(outputs {output: result + per_sport}, overall). Outputs with no rows are reported as not run."""
    ex = {x.strip().lower() for x in exclude_raters}
    rows = [r for r in rows if r["rater"].lower() not in ex]
    by = defaultdict(list)
    for r in rows: by[r["output"]].append(r)
    res = {}
    for out, t in TARGETS.items():
        rs = by.get(out, [])
        if not rs:
            res[out] = dict(label=t["label"], ran=False, passed=False); continue
        d = _score(rs, t); d.update(label=t["label"], ran=True, target=t)
        per_r = defaultdict(set)
        for r in rs: per_r[r["rater"]].add(r["match"])
        d["raters_under_3_matches"] = sorted(k for k, v in per_r.items() if len(v) < 3)
        d["per_sport"] = {s: _score([r for r in rs if r["sport"] == s], t) for s in sorted({r["sport"] for r in rs})}
        kt = defaultdict(int)
        for r in rs:
            if r["score"] < 1 and r["killer"]: kt[r["killer"]] += 1
        d["killers_when_not_won"] = dict(sorted(kt.items(), key=lambda kv: -kv[1]))
        res[out] = d
    ran = [v for v in res.values() if v["ran"]]
    overall = "PASS" if ran and all(v["passed"] for v in res.values()) else "FAIL" if any(v["ran"] and not v["passed"] for v in res.values()) else "INCOMPLETE"
    return dict(outputs=res, overall=overall, excluded_raters=sorted(ex))


def format_blind(rep):
    L = [f"blind test vs SwingVision: {rep['overall']}  (Wilson bound = lower end of the two-sided 90% interval)"]
    for out, d in rep["outputs"].items():
        if not d["ran"]:
            L.append(f"  ----  {d['label']:<24} not run"); continue
        t, c = d["target"], d["checks"]
        L.append(f"  {'PASS' if d['passed'] else 'FAIL'}  {d['label']:<24} win {d['win_rate']:.1%} ({d['wins']:g}/{d['n']}, ties {d['ties']})  "
                 f"Wilson90 lower {d['wilson_lower_90']:.3f}  raters {d['raters']}")
        need = [f"win >= {t['win_rate']:.0%}" + ("" if c["win_rate"] else " ✗"), f"n >= {t['min_comparisons']}" + ("" if c["comparisons"] else " ✗"),
                f"raters >= {t['min_raters']}" + ("" if c["raters"] else " ✗"), "lower > 50%" + ("" if c["wilson"] else " ✗")]
        L.append("        needs " + ", ".join(need))
        for s, v in d["per_sport"].items():
            L.append(f"        {s or '?':<11} win {v['win_rate']:.1%} n {v['n']:<4} lower {v['wilson_lower_90']:.3f} {'pass' if v['passed'] else 'not yet'}")
        if d["raters_under_3_matches"]:
            L.append(f"        note: {len(d['raters_under_3_matches'])} rater(s) rated fewer than 3 matches (§9.1 asks for >= 3)")
    return "\n".join(L)


def synthetic_votes(seed=0, reel_rate=0.78, full_rate=0.68, raters=32):
    """Each rater rates one reel pair and one full-cut pair on 3 matches (one per sport): 96 + 96 comparisons."""
    import random
    rnd = random.Random(seed); rows = []
    sports = ("tennis", "padel", "pickleball")
    for out, p in (("reel", reel_rate), ("full_cut", full_rate)):
        for m, sport in enumerate(sports):
            for k in range(raters):
                u = rnd.random(); v = 1.0 if u < p - 0.05 else 0.5 if u < p + 0.05 else 0.0
                rows.append(dict(rater=f"r{k:02d}", output=out, match=f"{sport}_m{m + 1}", sport=sport, score=v,
                                 killer="" if v == 1 else rnd.choice(("cut_off", "missed_point", "camera"))))
    return rows
