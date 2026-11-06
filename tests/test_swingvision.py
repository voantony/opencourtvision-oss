"""SwingVision importer and audit, on a small synthetic workbook built at test time (no real export in the repo)."""
import pytest

openpyxl = pytest.importorskip("openpyxl")

from opencourtvision.events import Event, verify  # noqa: E402
from opencourtvision.importers import swingvision as sv  # noqa: E402
from opencourtvision.render.cuts import MIN_TAIL_S, build_segments, point_end  # noqa: E402

SHOT_COLS = ["Player", "Shot", "Type", "Stroke", "Spin", "Speed (MPH)", "Point", "Game", "Set", "Bounce Depth", "Bounce Zone",
             "Bounce Side", "Bounce (x)", "Bounce (y)", "Hit Depth", "Hit Zone", "Hit Side", "Hit (x)", "Hit (y)", "Hit (z)",
             "Direction", "Result", "Favorited", "Start Time", "Video Time"]
POINT_COLS = ["Point", "Game", "Set", "Serve State", "Match Server", "Host Game Score", "Guest Game Score", "Point Winner", "Detail",
              "Break Point", "Set Point", "Favorited", "Start Time", "Video Time", "Duration"]
CAP = 130.596954


def shot(player, n, typ, stroke, mph, pt, res, vt, bx, by, hx, hy, hz=1.2, direction="cross court"):
    near_hit = hy < 11.885
    return [player, n, typ, stroke, "Flat", mph, pt, 1, 1, "deep", "deuce", "near" if by < 11.885 else "far", bx, by, "deep", "deuce",
            "near" if near_hit else "far", hx, hy, hz, direction, res, False, "20:00:00", vt]


def build(path):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Settings"
    ws.append(["Start Time", "End Time", "Location", "Host Team", "Guest Team", "Points", "Games", "Sets"])
    ws.append(["19:00:00", "20:00:00", None, "Host A / Host B", "Guest A / Guest B", 4, 1, 1])
    ws.append([]); ws.append(["Note: xyz coordinates are in meters."])
    S = wb.create_sheet("Shots"); S.append(SHOT_COLS)
    rows = [
        # P1: serve fault, faster second serve (95 > 60), return, host volley into the net -> guest wins
        shot("Host A", 1, "first_serve", "Serve", 60.0, 1, "Out", 100.0, -1.0, 20.0, 1.0, -0.3, 2.4, "out wide"),
        shot("Host A", 1, "second_serve", "Serve", 95.0, 1, "In", 106.0, -2.0, 15.0, 1.0, -0.3, 2.4, "down the T"),
        shot("Guest A", 2, "second_return", "Forehand", 40.0, 1, "In", 107.0, 1.0, 5.0, -2.0, 24.0),
        shot("Host B", 3, "serve_plus_one", "Forehand Volley", 30.0, 1, "Net", 108.0, 0.0, 10.0, -2.0, 9.0),
        # P2: capped first serve In, return into the net -> service winner
        shot("Host A", 1, "first_serve", "Serve", CAP, 2, "In", 130.0, 2.0, 15.0, -1.0, -0.3, 2.4, "out wide"),
        shot("Guest B", 2, "first_return", "Backhand", 35.0, 2, "Net", 131.0, 1.0, 13.0, 2.0, 24.0),
        # P3: double fault, cap value again (3 repeats in total)
        shot("Host A", 1, "first_serve", "Serve", CAP, 3, "Out", 160.0, 3.0, 20.0, 1.0, -0.3, 2.4, "out wide"),
        shot("Host A", 1, "second_serve", "Serve", CAP, 3, "Net", 166.0, 0.0, 10.0, 1.0, -0.3, 2.4, "down the T"),
        # P4: rally ball marked In but bouncing 1.5 m outside the doubles sideline; no Detail
        shot("Host A", 1, "first_serve", "Serve", 70.0, 4, "In", 200.0, 2.0, 15.0, -1.0, -0.3, 2.4, "out wide"),
        shot("Guest A", 2, "first_return", "Forehand", 40.0, 4, "In", 201.0, 7.0, 6.0, 2.0, 24.0),
    ]
    for r in rows: S.append(r)
    P = wb.create_sheet("Points"); P.append(POINT_COLS)
    P.append([1, 1, 1, "second", "host", "0", "0", "guest", "Forehand Unforced Error", False, False, False, "19:01:46", 108.0, 3.0])
    P.append([2, 1, 1, "first", "host", "0", "15", "host", "Service Winner", False, False, False, "19:02:12", 132.0, 1.5])
    P.append([3, 1, 1, "second", "host", "15", "15", "guest", "Double Fault", False, False, False, "19:02:48", 168.0, 1.0])
    P.append([4, 1, 1, "first", "host", "15", "30", "host", None, False, False, False, "19:03:22", 202.0, 4.0])
    G = wb.create_sheet("Games"); G.append(["Game", "Set", "Server", "Host Set Score", "Guest Set Score", "Game Winner", "Start Time", "Video Time", "Duration"])
    st = wb.create_sheet("Stats"); st.append(["Stat Name", "Host Set 1", "Guest Set 1"])
    st.append(["Total Points", 4, 4]); st.append(["Service Winners", 0, 0]); st.append(["Aces", 0, 0])
    wb.save(path); return path


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    return sv.load(build(str(tmp_path_factory.mktemp("sv") / "synthetic.xlsx")))


def test_load_and_roles(match):
    assert len(match.shots) == 10 and len(match.points) == 4
    assert match.roles == {"Host A": "H1", "Host B": "H2", "Guest A": "G1", "Guest B": "G2"}
    assert match.shots[0]["speed_mph"] == 60.0 and match.points[3].get("detail") is None


def test_events_are_unverified_references(match):
    ev, meta = sv.to_events(match)
    contacts = [e for e in ev if e.type == "contact"]
    assert len(contacts) == 10 and sum(e.type == "point_end" for e in ev) == 4 and sum(e.type == "point_start" for e in ev) == 4
    assert all(e.source == "swingvision" and e.verified is False for e in ev)
    assert {e.player for e in contacts} <= {"H1", "H2", "G1", "G2"} and "Host A" not in str(meta)
    first = contacts[0]
    assert first.t == pytest.approx(100.0 + sv.SV_SHOT_LAG_S) and first.attrs["video_time"] == 100.0
    verify(ev, min_confidence=0.0, min_sources=0)
    assert not any(e.verified for e in ev)
    assert any("reference import" in x for x in first.evidence)
    d = first.to_dict(); assert d["source"] == "swingvision" and Event.from_dict(d).attrs["point"] == 1


def test_labels(match):
    ev, _ = sv.to_events(match, lag_s=0.0)
    lab = {e.attrs["video_time"]: e.label for e in ev if e.type == "contact"}
    assert str(lab[106.0].shot_type) == "second_serve" and str(lab[106.0].direction) == "T"
    assert str(lab[100.0].direction) == "wide" and lab[100.0].outcome is None          # a fault is not a point-ending error
    assert str(lab[108.0].shot_type) == "volley" and str(lab[108.0].side) == "forehand" and str(lab[108.0].outcome) == "error"
    assert str(lab[107.0].shot_type) == "return" and str(lab[107.0].position) == "far_deuce"
    assert str(lab[100.0].position) == "near_deuce"
    pe = [e for e in ev if e.type == "point_end"]
    assert str(pe[1].label.outcome) == "win" and pe[3].label is None


def test_audit_flags(match):
    rep = sv.audit(match, examples=20)
    c = {x["id"]: x for x in rep["checks"]}
    assert c["serve_speed"]["count"] == 4                           # 95 and three capped values
    assert c["speed_repeats"]["count"] == 1 and "cap" in c["speed_repeats"]["note"]
    assert c["second_faster"]["count"] == 1
    assert c["in_but_outside"]["count"] == 1
    assert c["no_detail"]["count"] == 1
    assert c["double_fault"]["count"] == 0 and c["ace_sw"]["count"] == 0
    assert c["score_replay"]["count"] == 0
    assert any("Service Winners" in e for e in c["stats_consistency"]["examples"])
    assert rep["summary"]["frame"]["serve_hit_y_median_near"] == -0.3


def test_point_endings(match):
    pe = sv.point_endings(match, short=1.5)
    # P1 ends 108+3=111, last contact 108+2=110 -> 1.0 s; P2 132+1.5 vs 133 -> 0.5; P3 168+1 vs 168 -> 1.0; P4 206 vs 203 -> 3.0
    assert pe["n"] == 4 and pe["n_short"] == 3 and pe["min"] == 0.5 and pe["max"] == 3.0
    seg = sv.point_endings(match, segments=[(100.0, 113.0)])
    assert seg["ours_matched"] == 1 and seg["ours_post_min"] == pytest.approx(3.0)


def test_compare_points(match):
    segs = [dict(idx=1, start_s=101.0, end_s=113.0, first_hit_s=102.0, hits=4, fault_like=0, changeover_after=0),
            dict(idx=2, start_s=128.0, end_s=135.0, first_hit_s=132.05, hits=2, fault_like=0, changeover_after=0),
            dict(idx=3, start_s=400.0, end_s=405.0, first_hit_s=401.0, hits=1, fault_like=1, changeover_after=0)]
    hits = [102.0, 108.03, 109.0, 110.0, 132.05, 133.5]
    c = sv.compare_points(match, segs, hits=hits, score_rows=[{"score_before": "?"}])
    assert c["matched"] == 2 and len(c["missed"]) == 2 and c["extra"] == 1 and c["extra_fault_like"] == 1
    assert c["contacts"]["within_80ms"] == 5 and c["score"]["our_rows_with_score"] == 0
    off = sv.estimate_offset(match, [t + 0.5 for t in (102.0, 108.0, 109.0, 110.0, 132.0, 133.0, 162.0, 168.0)], search=(-2, 2))
    assert off["offset"] == pytest.approx(0.5, abs=0.02)


def test_cli_writes_report_without_names(tmp_path):
    x = build(str(tmp_path / "s.xlsx")); out = tmp_path / "audit.md"
    sv.main([x, "--out", str(out)])
    md = out.read_text()
    assert "Serve speed > 90 mph" in md and "Points ending < 1.5 s" in md
    assert "Host A" not in md and "Guest B" not in md


def test_point_end_never_before_min_tail():
    assert point_end(10.0, tail=0.5) == 10.0 + MIN_TAIL_S
    assert point_end(10.0, ball_dead=13.0) == pytest.approx(14.5)     # t_dec + 1.5 (point default)
    assert point_end(10.0, ball_dead=10.5) == 12.5                    # t_last + 2.5 (point default)
    R = [dict(t0=5.0, t1=9.0, hits=3, amax=20, pause_before=10, pause_after=30, times=[5.0, 7.0, 9.0])]
    KF = [float(i) for i in range(0, 60)]
    segs = build_segments(R, 0.0, KF, 60.0, lead=1.0, tail=0.3)
    assert segs[0]["B"] >= 9.0 + MIN_TAIL_S
