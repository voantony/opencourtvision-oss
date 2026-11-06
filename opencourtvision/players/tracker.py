"""IDTracker: 4 stable player ids for a doubles match, point by point, with a cheap-first cue cascade.

Per point (frames of one point, in-play detections only, <= 2 per side):
 1. motion: detections are linked into tracklets per side by court position (Hungarian on predicted position, human
    speed gate). A step where keep vs swap is within `margin_m` metres (partners crossing / merging) cuts both
    tracklets, so motion never carries an identity through an ambiguous moment. With motion off, every detection is
    its own 1-frame tracklet.
 2. tracklets that overlap in time on one side must be different people, so each side's overlap graph is a chain of
    2-colourable components; a component has exactly two labellings (which partner is which).
 3. court: the changeover schedule says which team is on which side; without it, team-to-side is decided too.
 4. cues are summed in cascade order per component and the cascade stops once the margin between the two
    labellings is >= `tau`. Appearance cues are measured on at most `k_frames` frames per tracklet (expensive
    extractions are charged only for those). Role cues (server / receiver at the serve contact + the per-set serve
    rotation and receiving sides) apply to the tracklet holding the serve frame.
 5. after the point, slot templates (EMA) and the rotation memory are updated from confident components.

Frame dicts: t, point, serve, set, server ('A'/'B'), parity, rside, near_team, dets (list of dicts: xy, side, and the
cue features colour_box / colour_mask / osnet / head vectors, role_* + p_role_* on serve frames).

Appearance data (Ops V6, closed EU beta; privacy notice v2 §4): OSNet embeddings and colour histograms live in this object's
memory for ONE match only. The tracker never writes them anywhere, refuses to be pickled / copied / serialised, returns
only slot ids and margins, and `end_match()` (or leaving a `with IDTracker() as tr:` block) drops every template. A new
match needs a new tracker; nothing is carried across matches, so there is no "recognise this person" profile.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .cues import cosine, hellinger

SLOTS = ("A1", "A2", "B1", "B2")
TEAM = {"A1": "A", "A2": "A", "B1": "B", "B2": "B"}
APPEARANCE_KEYS = ("colour_box", "colour_mask", "osnet", "head")   # per-match, in-memory only (Ops V6)
APPEAR = {"colour_box": hellinger, "colour_mask": hellinger, "osnet": cosine, "head": cosine}
ROLE = ("role_pos", "role_pose", "role_posenet")
ALL_CUES = ("court", "motion", "role_pos", "colour_box", "colour_mask", "osnet", "head", "role_pose", "role_posenet")
NEEDS = {"colour_box": ("colour_box",), "colour_mask": ("vision", "colour_mask"), "osnet": ("osnet",), "head": ("osnet", "head"),
         "role_pose": ("vision",), "role_posenet": ("vision", "posenet")}
DEFAULT_SCALE = {"colour_box": 0.15, "colour_mask": 0.15, "osnet": 0.2, "head": 0.4, "role": 1.0}


@dataclass
class Tracklet:
    side: int
    items: list = field(default_factory=list)       # (frame_pos_in_point, det_index)
    xy: np.ndarray | None = None
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t: float = 0.0
    alive: bool = True


class IDTracker:
    def __init__(self, cues=ALL_CUES, order=None, tau=None, weights=None, scale=None, unit_ms=None, alpha=0.2,
                 upd_margin=0.5, vmax=8.0, margin_m=0.8, k_frames=3):
        self.cues = tuple(c for c in (order or ALL_CUES) if c in cues)
        self.tau, self.alpha, self.upd_margin, self.vmax, self.margin_m, self.k = tau, alpha, upd_margin, vmax, margin_m, k_frames
        self.w = {c: 1.0 for c in ALL_CUES} | (weights or {})
        self.scale = DEFAULT_SCALE | (scale or {})
        self.unit_ms = unit_ms or {}
        self.tmpl = {s: {} for s in SLOTS}
        self.anchor = {}
        self.rot, self.rcv = {}, {}
        self.recorder = None                           # f(point_frames, comps, cost_rows) for learned fusion
        self.ms_feat, self.ms_track, self.n_frames = 0.0, 0.0, 0
        self.cue_used = {c: 0 for c in ALL_CUES}; self.decisions = 0

    # ------------------------------------------------------------------ appearance data: in memory, one match (Ops V6)
    def __reduce_ex__(self, protocol):
        raise TypeError("IDTracker holds per-match appearance templates (OSNet / colour); it is never pickled, copied or saved")

    def __getstate__(self):
        raise TypeError("IDTracker holds per-match appearance templates (OSNet / colour); it is never pickled, copied or saved")

    def end_match(self):
        """Drop every appearance template, tap anchor and rotation memory. Call when the match ends."""
        self.tmpl = {s: {} for s in SLOTS}
        self.anchor = {}
        self.rot, self.rcv = {}, {}
        self.recorder = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.end_match()
        return False

    # ------------------------------------------------------------------ taps
    def init(self, frame, mapping):
        # the taps are ground truth: they also pin the labelling of the tracklets that contain them (anchor), so the
        # tap point's decisions are certain and seed the templates and the serve-rotation memory
        self.anchor = {(frame.get("point"), round(float(frame.get("t", 0.0)), 3), i): s for i, s in mapping.items()}
        for i, s in mapping.items():
            d = frame["dets"][i]
            self.tmpl[s] = {c: np.asarray(d[c], float).copy() for c in APPEAR if d.get(c) is not None}
            self._remember(frame, d, s)

    def _mem(self, frame, team, role, create=False):
        key = (frame["set"], team)
        if role == "srv" and team == frame["server"]:
            store, k = self.rot, frame["parity"]
        elif role == "rcv" and team != frame["server"]:
            store, k = self.rcv, frame["rside"]
        else:
            return None
        if create:
            return store.setdefault(key, {}), k
        return store.get(key, {}).get(k, {})

    def _remember(self, frame, d, s):
        role = next((d.get(c) for c in self.cues if c in ROLE and d.get(c)), None)
        if not frame.get("serve") or role is None:
            return
        got = self._mem(frame, TEAM[s], role, create=True)
        if got is not None:
            store, k = got; other = [x for x in SLOTS if TEAM[x] == TEAM[s] and x != s][0]
            for kk, ss in ((k, s), (1 - k, other)):
                store.setdefault(kk, {}).setdefault(ss, 0); store[kk][ss] += 1

    # ------------------------------------------------------------------ 1. motion tracklets
    def _tracklets(self, F):
        out = []
        for side in (0, 1):
            act = []
            for p, f in enumerate(F):
                dis = [i for i, d in enumerate(f["dets"]) if d["side"] == side]
                if "motion" not in self.cues:
                    out += [Tracklet(side, [(p, i)]) for i in dis]; continue
                act = [tk for tk in act if f["t"] - tk.t <= 1.0]
                X = {i: np.asarray(f["dets"][i]["xy"], float) for i in dis}
                best, alt = None, None
                for k in range(min(len(act), len(dis)), -1, -1):
                    for ts in itertools.permutations(range(len(act)), k):
                        for ds in itertools.combinations(dis, k):
                            ok = True; c = 0.0
                            for ti, di in zip(ts, ds):
                                tk = act[ti]; dt = f["t"] - tk.t; dist = float(np.linalg.norm(X[di] - (tk.xy + tk.v * min(dt, 0.5))))
                                if dist > self.vmax * dt + 0.7:
                                    ok = False; break
                                c += dist
                            if ok:
                                cand = (c, dict(zip(ds, ts)))
                                if best is None or cand[0] < best[0]:
                                    best, alt = cand, best
                                elif alt is None or cand[0] < alt[0]:
                                    alt = cand
                    if best is not None:
                        break
                m = best[1] if best else {}
                ambiguous = best is not None and alt is not None and len(alt[1]) == len(m) and len(m) == 2 and alt[0] - best[0] < self.margin_m
                if ambiguous:
                    m = {}; act = []
                new = []
                for di in dis:
                    if di in m:
                        tk = act[m[di]]; dt = f["t"] - tk.t
                        tk.v = 0.5 * tk.v + 0.5 * (X[di] - tk.xy) / max(dt, 1e-3)
                    else:
                        tk = Tracklet(side); out.append(tk); tk.v = np.zeros(2)
                    tk.xy, tk.t = X[di], f["t"]; tk.items.append((p, di)); new.append(tk)
                act = new if ambiguous else new + [a for a in act if all(a is not n for n in new)]
        return out

    # ------------------------------------------------------------------ 2. components
    @staticmethod
    def _components(tks):
        """per side: tracklets overlapping in time -> 2-colouring components [(tracklet_indices, colour bits)]."""
        comps = []
        for side in (0, 1):
            idx = [i for i, t in enumerate(tks) if t.side == side]
            span = {i: {p for p, _ in tks[i].items} for i in idx}
            adj = {i: [j for j in idx if j != i and span[i] & span[j]] for i in idx}
            seen = set()
            for i in idx:
                if i in seen:
                    continue
                col = {i: 0}; stack = [i]; seen.add(i)
                while stack:
                    a = stack.pop()
                    for b in adj[a]:
                        if b not in col:
                            col[b] = 1 - col[a]; seen.add(b); stack.append(b)
                comps.append((side, col))
        return comps

    # ------------------------------------------------------------------ 3./4. costs
    def _sample(self, tk):
        it = tk.items
        if len(it) <= self.k:
            return it
        return [it[int(round(j))] for j in np.linspace(0, len(it) - 1, self.k)]

    def _feat(self, F, tk, cue, charged):
        vals = []
        for p, i in self._sample(tk):
            d = F[p]["dets"][i]
            if d.get(cue) is not None:
                vals.append(np.asarray(d[cue], float))
            for e in NEEDS.get(cue, ()):
                charged.add((p, i, e))
        return np.mean(vals, 0) if vals else None

    def _cost(self, cue, F, tk, slot, cache, charged):
        if cue in APPEAR:
            key = (id(tk), cue)
            if key not in cache:
                cache[key] = self._feat(F, tk, cue, charged)
            v = cache[key]
            if v is None or cue not in self.tmpl[slot]:
                return 0.0
            return APPEAR[cue](self.tmpl[slot][cue], v) / self.scale[cue]
        if cue in ROLE:
            c = 0.0
            for p, i in tk.items:
                f = F[p]; d = f["dets"][i]
                if not f.get("serve") or d.get(cue) is None:
                    continue
                for e in NEEDS.get(cue, ()):
                    charged.add((p, i, e))
                mem = self._mem(f, TEAM[slot], d[cue])
                if not mem:
                    continue
                best = max(mem, key=mem.get); others = sum(v for k, v in mem.items() if k != best)
                if mem[best] - others >= 1 and slot != best:
                    c += 2.0 * max(0.0, 2 * float(d.get("p_" + cue, 0.5)) - 1.0) / self.scale["role"]
            return c
        return 0.0

    def _comp_cost(self, F, tks, side_team, comp, cue, cache, charged):
        """cost of labelling 0 and labelling 1 of a component for one cue."""
        side, col = comp; team = side_team[side]; slots = (team + "1", team + "2")
        out = []
        for lab in (0, 1):
            out.append(sum(self._cost(cue, F, tks[i], slots[c ^ lab], cache, charged) for i, c in col.items()))
        return out

    def _anchor_cost(self, F, tks, side_team, comp):
        acc = np.zeros(2)
        if not self.anchor:
            return acc
        side, col = comp; team = side_team[side]; slots = (team + "1", team + "2")
        for i, c in col.items():
            for p, di in tks[i].items:
                s = self.anchor.get((F[p].get("point"), round(float(F[p].get("t", 0.0)), 3), di))
                if s is None:
                    continue
                for lab in (0, 1):
                    acc[lab] += 0.0 if slots[c ^ lab] == s else 1e3
        return acc

    # ------------------------------------------------------------------ main
    def point(self, F):
        """assign ids for one point's frames; returns list (per frame) of {det_index: slot} and per-det margins."""
        t0 = time.perf_counter()
        tks = self._tracklets(F); comps = self._components(tks)
        cues = [c for c in self.cues if c not in ("court", "motion")]
        near = F[0]["near_team"]; other = "B" if near == "A" else "A"
        options = [{0: near, 1: other}] if "court" in self.cues else [{0: near, 1: other}, {0: other, 1: near}]
        cache, charged = {}, set(); best = None
        for side_team in options:
            total = 0.0; labs = []
            for comp in comps:
                acc = self._anchor_cost(F, tks, side_team, comp); used = []
                for c in cues or [None]:
                    if c is not None:
                        acc += self.w.get(c, 1.0) * np.array(self._comp_cost(F, tks, side_team, comp, c, cache, charged)); used.append(c)
                    if self.tau is not None and abs(acc[0] - acc[1]) >= self.tau:
                        break
                lab = int(acc[1] < acc[0]); labs.append((lab, float(abs(acc[0] - acc[1])), used)); total += acc[lab]
            if best is None or total < best[0]:
                best = (total, side_team, labs)
        _, side_team, labs = best
        if self.recorder is not None:
            rows = [{c: self._comp_cost(F, tks, side_team, comp, c, {}, set()) for c in cues} for comp in comps]
            self.recorder(F, tks, comps, side_team, rows)
        preds = [{} for _ in F]; margins = [{} for _ in F]
        for comp, (lab, mg, used) in zip(comps, labs):
            side, col = comp; team = side_team[side]
            self.decisions += 1
            for c in used:
                self.cue_used[c] += 1
            for i, c in col.items():
                s = f"{team}{1 + (c ^ lab)}"
                for p, di in tks[i].items:
                    preds[p][di] = s; margins[p][di] = mg
        self.ms_track += 1000 * (time.perf_counter() - t0)
        self.ms_feat += sum(self.unit_ms.get(e, 0.0) for _, _, e in charged); self.n_frames += len(F)
        # 5. updates
        for comp, (lab, mg, _) in zip(comps, labs):
            if mg < self.upd_margin:
                continue
            side, col = comp; team = side_team[side]
            for i, c in col.items():
                s = f"{team}{1 + (c ^ lab)}"
                for cue in APPEAR:
                    v = cache.get((id(tks[i]), cue))
                    if v is None:
                        v = self._feat(F, tks[i], cue, set())
                    if v is None:
                        continue
                    self.tmpl[s][cue] = v.copy() if cue not in self.tmpl[s] else (1 - self.alpha) * self.tmpl[s][cue] + self.alpha * v
                for p, di in tks[i].items:
                    if F[p].get("serve"):
                        self._remember(F[p], F[p]["dets"][di], s)
        return preds, margins

    def run(self, frames):
        out, i = [], 0
        while i < len(frames):
            j = i
            while j < len(frames) and frames[j]["point"] == frames[i]["point"]:
                j += 1
            out += self.point(frames[i:j])[0]; i = j
        return out

    @property
    def ms_per_frame(self):
        return (self.ms_feat + self.ms_track) / max(1, self.n_frames)
