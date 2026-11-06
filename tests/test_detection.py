import json

from benchmarks.run_eval import match_hits
from opencourtvision.detection.audio import detect_hits, group_rallies, load_audio, loudness
from opencourtvision.detection.match_start import estimate_match_start, warmup_cutoff


def _detect(path):
    sr, x = load_audio(path)
    le, bg, _ = loudness(sr, x)
    t, amp = detect_hits(le, bg)
    return t, group_rallies(t, amp)


def test_hits_clean_precision_recall_timing(clips):
    gt = json.load(open(clips["hits_clean"]["gt"]))
    t, R = _detect(clips["hits_clean"]["mp4"])
    tp, fp, fn, errs = match_hits(list(t), gt["hits"], 0.05)
    assert tp / (tp + fp) >= 0.95 and tp / (tp + fn) >= 0.95
    assert max(errs) <= 0.02
    assert len(R) == len(gt["rallies"])


def test_wav_and_mp4_agree(clips):
    t_wav, _ = _detect(clips["hits_clean"]["wav"])
    t_mp4, _ = _detect(clips["hits_clean"]["mp4"])
    assert len(t_wav) == len(t_mp4)


def test_warmup_cutoff_finds_first_serve(clips):
    gt = json.load(open(clips["match_sim"]["gt"]))
    _, R = _detect(clips["match_sim"]["wav"])
    start, ev = warmup_cutoff(R)
    assert abs(start - gt["match_start_s"]) < 1.0
    assert ev["confidence"] == "high"
    assert warmup_cutoff(R, override=12.5) == (12.5, {"method": "override"})


def test_rhythm_estimator_runs(clips):
    res = estimate_match_start(clips["match_sim"]["wav"])
    assert res["n_hits"] > 200 and res.get("estimate_s") is not None
