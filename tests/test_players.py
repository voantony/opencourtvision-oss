"""4-player ID tracker: rules, deduction from 2 taps, tracklet cascade, rules truth, metrics (synthetic match only)."""
import numpy as np
import pytest

from opencourtvision.players import geometry as G
from opencourtvision.players.cues import COLOUR_DIM, colour_hist, hellinger, pose_norm, serve_pose_score
from opencourtvision.players.deduce import deduce_ids
from opencourtvision.players.inplay import select
from opencourtvision.players.metrics import evaluate
from opencourtvision.players.rules import end_flips, near_team, serve_parity, serve_roles
from opencourtvision.players.synthetic import make_frames, match_points
from opencourtvision.players.tracker import IDTracker
from opencourtvision.players.truth import build_serve_truth


def _pt(sets, games, pts=(0, 0), tb=False):
    return dict(score_before=dict(sets=list(sets), games=list(games), pts=list(pts), tiebreak=tb))


def test_end_flips_odd_games_and_set_boundary():
    # set 1 ends 6-1 (7 games, odd): ends change after games 1,3,5,7 -> set 2 starts with 4 flips (same end as start)
    P = [_pt((0, 0), (0, 0)), _pt((0, 0), (1, 0)), _pt((0, 0), (1, 1)), _pt((0, 0), (6, 0)), _pt((1, 0), (0, 0)), _pt((1, 0), (1, 0))]
    assert end_flips(P) == [0, 1, 1, 3, 4, 5]
    assert near_team(P, True)[:2] == ["A", "B"]
    tb = [_pt((0, 0), (6, 6), (0, 0), True), _pt((0, 0), (6, 6), (3, 3), True)]
    assert end_flips(tb)[1] - end_flips(tb)[0] == 1


def test_serve_parity_and_roles():
    assert [serve_parity(_pt((0, 0), g)) for g in ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2))] == [0, 0, 1, 1, 0]
    # near server deep at x>mid; receiver far on the diagonal (x<mid)
    si, ri, cs, cr = serve_roles([[7.0, 9.0], [7.5, -0.4]], [[3.0, 24.0], [8.0, 15.0]])
    assert si == 1 and ri == 0 and cs > 0.9 and cr > 0.9


def test_deduce_two_taps():
    dets = [dict(side=0), dict(side=0), dict(side=1), dict(side=1)]
    m, st = deduce_ids(dets, [(0, "A1"), (3, "B1")])
    assert st == "ok" and m == {0: "A1", 1: "A2", 3: "B1", 2: "B2"}
    m, st = deduce_ids(dets, [(0, "A1"), (1, "A2")])
    assert st == "anon_opp" and {m[2], m[3]} == {"B1", "B2"}
    m, st = deduce_ids(dets, [(0, "A1")])
    assert st == "anon_opp" and m[1] == "A2"
    assert deduce_ids(dets, [(0, "A1"), (1, "B1")])[1] == "defer:taps_conflict"
    assert deduce_ids(dets + [dict(side=1)], [(0, "A1")])[1].startswith("defer:side1")
    assert deduce_ids(dets, [])[1] == "anon_all"


def test_cues_and_geometry():
    img = np.zeros((100, 40, 3), np.uint8); img[20:50] = (0, 0, 255); img[50:75] = (255, 255, 255)
    h = colour_hist(img); assert h.shape == (COLOUR_DIM,) and abs(np.linalg.norm(h) - np.sqrt(2)) < 1e-4
    img2 = img.copy(); img2[20:50] = (255, 0, 0)
    assert hellinger(h, colour_hist(img2)) > 0.3 and hellinger(h, colour_hist(img)) < 1e-6
    kp = np.zeros((17, 3)); kp[:, 2] = 1; kp[[5, 6], 1] = 0.3; kp[[11, 12], 1] = 0.6; kp[[9, 10], 1] = 0.5; kp[9, 1] = 0.0
    assert serve_pose_score(kp) == pytest.approx(1.0) and pose_norm(kp).shape == (51,)
    kp[11:13, 2] = 0
    assert pose_norm(kp) is None
    assert len(G.nms([[0, 0, 10, 30, 0.9], [1, 1, 9, 20, 0.8]], 0.6, contain=0.8)) == 1
    assert select([dict(side=0)] * 3 + [dict(side=1)], [0.9, 0.1, 0.5, 0.2]) == [0, 2, 3]


@pytest.fixture(scope="module")
def synth():
    from opencourtvision.players.cues import pose_norm as pn
    from opencourtvision.players.cues import serve_pose_score as sps
    P = match_points(seed=1); frames, _ = make_frames(P, seed=1, spectators=0)
    nt = dict(zip([p["idx"] for p in P], near_team(P, True)))
    for f in frames:
        f["near_team"] = nt[f["point"]]
        for d in f["dets"]:
            d["serve_pose"] = sps(d["kp"]); d["pose_vec"] = pn(d["kp"])
    return P, frames


def _acc(frames, cues, **kw):
    tr = IDTracker(cues=cues, **kw); f0 = frames[3]; tr.init(f0, {i: d["_true"] for i, d in enumerate(f0["dets"])})
    preds = tr.run(frames); ok = n = 0
    for f, pr in zip(frames[4:], preds[4:]):
        for i, s in pr.items():
            n += 1; ok += s == f["dets"][i]["_true"]
    return ok / n, tr


def test_tracker_cues_on_synthetic(synth):
    _, frames = synth
    full, tr = _acc(frames, ("court", "motion", "colour_box", "osnet"))
    assert full > 0.98 and tr.ms_per_frame >= 0
    base, _ = _acc(frames, ("court", "motion"))
    assert base < 0.75                                       # court + motion cannot tell partners apart across points
    casc, trc = _acc(frames, ("court", "motion", "colour_box", "osnet"), tau=1.0)
    assert casc > 0.95 and trc.cue_used["osnet"] < tr.cue_used["osnet"]


def test_rules_truth_matches_synthetic(synth):
    P, frames = synth
    keep = [[i for i, d in enumerate(f["dets"]) if d["_true"]] for f in frames]
    serve_idx = {f["point"]: i for i, f in enumerate(frames) if f["serve"]}
    truth, rep = build_serve_truth(frames, keep, P, serve_idx)
    assert rep["points_with_id_truth"] >= 0.8 * len(P) and rep["schedule_agreement"] == 1.0
    agree = [frames[fi]["dets"][d]["_true"] == s for fi, t in truth.items() for d, s in t.items()]
    assert np.mean(agree) > 0.97
    preds = [{d: frames[fi]["dets"][d]["_true"] for d in keep[fi]} for fi in range(len(frames))]
    r = evaluate(frames, preds, truth)
    assert r["id_acc"] > 0.97 and r["false_per_100_frames"] == 0.0


def test_merge_far_drops_cut_tile_boxes():
    roi = (1000, 300, 2000, 540)                      # bottom edge at y = 840
    full = [[1800, 600, 1900, 1000, 0.9], [2500, 500, 2540, 620, 0.5]]      # near net player (feet below roi), far player
    far = [[1802, 605, 1898, 840, 0.95], [2498, 498, 2542, 622, 0.9]]       # cut copy of the near player, far player
    m = G.merge_far(full, far, roi)
    assert [1800, 600, 1900, 1000, 0.9] in m and [1802, 605, 1898, 840, 0.95] not in m
    assert [2498, 498, 2542, 622, 0.9] in m and [2500, 500, 2540, 620, 0.5] not in m


def test_side_roles_with_a_missing_near_player():
    from opencourtvision.players.rules import side_roles
    srv, rcv = side_roles([[3.0, 24.3]], [[3.5, 7.0]])          # far server visible; only the near net player is seen
    assert srv == [("main", 1.0)] and rcv == [("partner", 1.0)]
    srv, rcv = side_roles([[7.0, 14.0], [3.0, 24.0]], [[8.0, -0.5]])
    assert srv[1][0] == "main" and rcv == [("main", 1.0)]
