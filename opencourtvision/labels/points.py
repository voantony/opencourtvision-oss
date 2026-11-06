"""Point-level ground truth from tools/labeler (schema "ocv.point_labels.v1") -> scoring decoder inputs and training rows.

A labels file has one entry per video segment, in time order:
  {idx, start, end, kind: "point" | "non_point" | null (unlabeled), winner: "A" | "B" | null,
   winner_src: "user" | "user_accepted_suggestion" | null, server, server_src: "rules" | "user", side: "deuce" | "ad" | null,
   score_before: {sets, games, pts, tiebreak, call}, notes, server_override, sugg_* / sv_* (seed hints, NOT ground truth)}
plus match settings (teams, mine, first_server, preset) and `decoded` (score from the labels by the rules).
Teams are "A" / "B" as in scoring/decoder.py. Only `kind` and `winner` are ground truth; server/side/score are derived
from them by the rules (plus user server overrides) and re-checked here by `verify()`.
"""
import json
from dataclasses import dataclass, field, replace

from ..scoring.formats import format_from_settings
from ..scoring.rules import Obs, PointEvidence, advance, initial_state, score_call, serve_side

SCHEMA = "ocv.point_labels.v1"


@dataclass
class LabeledSegment:
    idx: int
    start: float
    end: float
    kind: str = None
    winner: str = None
    winner_src: str = None
    server_override: str = None
    notes: str = ""
    server: str = None
    side: str = None
    score_before: dict = None
    favorite: bool = False
    fav_seen: bool = False           # shown in the favourites pass (or starred): a non-favourite is a negative only if seen
    extra: dict = field(default_factory=dict)

    @property
    def choice(self):
        """Decoder choice: 'A' / 'B' / 'skip', or None if unlabeled."""
        return "skip" if self.kind == "non_point" else self.winner if self.kind == "point" else None


@dataclass
class PointLabels:
    video: dict
    match: dict
    segments: list
    decoded: dict
    labeler: dict

    @property
    def fmt(self):
        return format_from_settings({"preset": "advantage", **self.match})

    @property
    def complete(self):
        return all(s.choice is not None for s in self.segments)

    def states(self):
        """Score state before each segment, from the labels (unlabeled segments count as skips)."""
        fmt = self.fmt
        st = initial_state(fmt, self.match.get("first_server") or "A")
        out = []
        for s in self.segments:
            if s.server_override and not st.done:
                st = replace(st, server=s.server_override)
            out.append(st)
            c = s.choice
            if c in ("A", "B") and not st.done:
                st = advance(st, c, fmt)
        out.append(st)
        return out

    def verify(self):
        """Re-derive server/side/score for every segment and compare with what the labeler exported. Returns mismatches."""
        fmt, bad = self.fmt, []
        for s, st in zip(self.segments, self.states()):
            if s.score_before is None:
                continue
            exp = dict(sets=list(st.sets), games=list(st.games), pts=list(st.pts), call=score_call(st, fmt))
            got = {k: s.score_before.get(k) for k in exp}
            if got != exp or s.server != st.server or s.side != serve_side(st, fmt):
                bad.append((s.idx, dict(got, server=s.server, side=s.side), dict(exp, server=st.server, side=serve_side(st, fmt))))
        return bad

    def final_score(self):
        """'6-1 6-2' style score of the labeled match."""
        sets, prev = [], None
        for st in self.states():
            if prev is not None and st.sets != prev.sets:
                g = list(prev.games)
                g[0 if st.sets[0] > prev.sets[0] else 1] += 1
                sets.append(f"{g[0]}-{g[1]}")
            prev = st
        if not prev.done:
            sets.append(f"{prev.games[0]}-{prev.games[1]}")
        return " ".join(sets)

    def winners(self):
        """Decoder truth sequence ('A'/'B'/'skip'; None for unlabeled)."""
        return [s.choice for s in self.segments]

    def to_evidence(self, server_acc=None):
        """PointEvidence per segment with the labels as hard confirmations (unlabeled -> no confirmation).
        server_acc: also attach the labeled server as an observation with this accuracy (e.g. for decoder tests)."""
        ev = []
        for s in self.segments:
            kw = dict(confirmed=s.choice, meta=dict(start=s.start, end=s.end, idx=s.idx))
            if server_acc and s.server:
                kw["server"] = Obs(s.server, server_acc)
            ev.append(PointEvidence(**kw))
        return ev

    def training_rows(self, video_id=None):
        """Flat per-segment rows for model training (labeled segments only)."""
        vid = video_id or self.video.get("name")
        mine = self.match.get("mine", "A")
        rows = []
        for s, st in zip(self.segments, self.states()):
            if s.choice is None:
                continue
            rows.append(dict(video=vid, date=self.match.get("date"), idx=s.idx, start=s.start, end=s.end,
                             is_point=s.kind == "point", winner=s.winner, winner_is_mine=(s.winner == mine) if s.winner else None,
                             server=st.server, side=serve_side(st, self.fmt), call=score_call(st, self.fmt),
                             sets=list(st.sets), games=list(st.games), pts=list(st.pts), tiebreak=bool(st.tb),
                             label_src=s.winner_src, server_src="user" if s.server_override else "rules", notes=s.notes))
        return rows


    def favorites(self):
        return [s for s in self.segments if s.favorite]

    def highlight_rows(self, sv_points=None, settings=None, sv_offset=2.0, weights=None, user=None):
        """Favourite-vs-not rows for training the highlight ranker (story/highlight.py), one per labelled segment.
        Points get the ranker's features: the score context comes from the LABELS (true winners / score), rally features
        (shots, span, fast exchanges, net exchange, ending, reaction dB) from the matched SwingVision point when given.
        X segments (high fives, celebrations) get is_point=False and no ranker features. `seen` = shown in the favourites
        pass or starred: only seen non-favourites are reliable negatives."""
        from ..story.config import MatchSettings, UserConfig
        from ..story.highlight import DEFAULT_WEIGHTS, PointInput, great_lost_points, personal_scores, score_points
        weights = weights or DEFAULT_WEIGHTS
        segs, states = self.segments, self.states()
        svmap = match_sv_points(segs, sv_points, sv_offset) if sv_points else {}
        host = self.match.get("mine", "A")
        side = {host: "host", ("B" if host == "A" else "A"): "guest"}
        inputs, where = [], {}
        n = 0
        for i, s in enumerate(segs):
            if s.kind != "point":
                continue
            st = states[i]; n += 1
            q = svmap.get(i) or {}
            hp, gp = (st.pts if host == "A" else st.pts[::-1])
            g = tuple(st.games if host == "A" else st.games[::-1]); ss = tuple(st.sets if host == "A" else st.sets[::-1])
            d = dict(point=n, server=side[st.server], winner=side[s.winner], host_pts=hp, guest_pts=gp, games_before=g,
                     sets_before=ss, t_last=s.end, set=ss[0] + ss[1] + 1)
            for k in ("rally_shots", "rally_span_s", "fast_exchanges", "net_exchange", "ending", "detail", "react_db", "react_dur",
                      "high_difficulty"):
                if k in q:
                    d[k] = q[k]
            where[len(inputs)] = i
            inputs.append(PointInput.from_dict(d))
        st_ = settings or MatchSettings(user_team="host")
        sl = score_points(inputs, st_, weights) if inputs else []
        great = set(great_lost_points(sl, weights)) if sl else set()
        pers = personal_scores(sl, great, (user or UserConfig()).loss_weight, weights) if sl else {}
        scored = {}
        for k, r in enumerate(sl):
            pr = r["pressure"]
            scored[where[k]] = dict(r, great_lost=r["point"] in great, personal=pers.get(r["point"]),
                                    game_state=pr.get("game_state"), break_point=bool(pr.get("break_point")),
                                    set_point=bool(pr.get("set_point")), match_point=bool(pr.get("match_point")),
                                    game_point_for=pr.get("game_point_for"))
        feats = ("pq", "M", "content", "g_rally", "g_out", "g_react", "clip_score", "coaching_value", "rally_difficulty",
                 "v_rally", "v_dur", "v_fast", "v_netx", "v_react", "rally_shots", "ending", "lost", "great_lost", "personal",
                 "game_state", "break_point", "set_point", "match_point", "game_point_for", "detail", "server")
        rows = []
        for i, s in enumerate(segs):
            if s.choice is None:
                continue
            r = dict(video=self.video.get("name"), idx=s.idx, start=s.start, end=s.end, is_point=s.kind == "point",
                     favorite=s.favorite, seen=s.fav_seen, winner=s.winner, notes=s.notes, has_sv=i in svmap)
            if i in scored:
                r["point"] = scored[i]["point"]
                r.update({k: scored[i].get(k) for k in feats})
            rows.append(r)
        return rows


def match_sv_points(segs, sv, offset=2.0, tol=6.0):
    """SwingVision points -> {segment index: point} by SV last-shot time + offset; labelled points preferred, one per segment."""
    out, used = {}, set()
    for q in sv or []:
        t = float(q["t_last"]) + offset

        def gap(s):
            return 0.0 if s.start <= t <= s.end else min(abs(t - s.start), abs(t - s.end))
        c = sorted((gap(s), i) for i, s in enumerate(segs) if i not in used)
        pts = [(g, i) for g, i in c if g <= tol and segs[i].kind == "point"]
        best = pts[0] if pts else (c[0] if c and c[0][0] <= tol else None)
        if best:
            used.add(best[1]); out[best[1]] = q
    return out


def ranker_vs_favorites(rows, ns=(5, 10, 20)):
    """How well the current ranker's top-N (by pq, points only) matches the user's favourites among SEEN points."""
    pts = [r for r in rows if r["is_point"] and r.get("pq") is not None and r["seen"]]
    fav = {r["idx"] for r in pts if r["favorite"]}
    if not fav:
        return dict(n_seen=len(pts), n_fav=0)
    order = [r["idx"] for r in sorted(pts, key=lambda r: (-r["pq"], r["idx"]))]
    out = dict(n_seen=len(pts), n_fav=len(fav), favorite_ranks=sorted(order.index(i) + 1 for i in fav))
    for n in ns:
        top = set(order[:n])
        out[f"precision@{n}"] = len(top & fav) / n
        out[f"recall@{n}"] = len(top & fav) / len(fav)
    hits, ap = 0, 0.0
    for k, i in enumerate(order, 1):
        if i in fav:
            hits += 1; ap += hits / k
    out["average_precision"] = ap / len(fav)
    out["random_ap"] = len(fav) / len(pts)
    return out


_KNOWN = {"idx", "start", "end", "kind", "winner", "winner_src", "server_override", "notes", "server", "side", "score_before",
          "favorite", "fav_seen"}


def parse_point_labels(obj):
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} file (schema={obj.get('schema')!r})")
    segs = []
    for i, p in enumerate(obj["points"]):
        kind = p.get("kind") or ("point" if p.get("winner") else None)
        if kind not in (None, "point", "non_point"):
            raise ValueError(f"segment {i + 1}: bad kind {kind!r}")
        if kind == "point" and p.get("winner") not in ("A", "B"):
            raise ValueError(f"segment {i + 1}: point without winner A/B")
        segs.append(LabeledSegment(idx=p.get("idx", i + 1), start=float(p["start"]), end=float(p["end"]), kind=kind,
                                   winner=p.get("winner") if kind == "point" else None, winner_src=p.get("winner_src"),
                                   server_override=p.get("server_override"), notes=p.get("notes") or "",
                                   server=p.get("server"), side=p.get("side"), score_before=p.get("score_before"),
                                   favorite=bool(p.get("favorite")), fav_seen=bool(p.get("fav_seen") or p.get("favorite")),
                                   extra={k: v for k, v in p.items() if k not in _KNOWN}))
    for a, b in zip(segs, segs[1:]):
        if b.start < a.start:
            raise ValueError(f"segments out of order at {b.idx}")
    return PointLabels(video=obj.get("video") or {}, match=obj.get("match") or {}, segments=segs,
                       decoded=obj.get("decoded") or {}, labeler=obj.get("labeler") or {})


def load_point_labels(path):
    with open(path) as f:
        return parse_point_labels(json.load(f))
