#!/usr/bin/env python3
"""Build points.json for export_clips from audio point segmentation.

Uses the repo's editloop audio pipeline (rallies_only / smart). Without ball
tracks, pw_rules abstain, so every point gets p_a=0.5 / tier=abstain. Decoder
ask_order still runs at export time.

  python tools/labeling/make_candidates.py --video match.mp4 --out points.json \
      --sport padel --preset golden_point --license "CC BY" \
      [--max-seconds 1200] [--video-id ID] [--url URL] [--attribution TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencourtvision.detection.audio import detect_hits, extract_wav, group_rallies, loudness, read_wav
from opencourtvision.detection.match_start import warmup_cutoff
from opencourtvision.editloop.build import flags as edit_flags
from opencourtvision.editloop.modes import rallies_only
from opencourtvision.ffmpeg import duration as video_duration


def build(video: Path, work: Path, sport: str, preset: str, license_: str,
          video_id: str, url: str, attribution: str, max_seconds: float | None,
          first_server: str = "A", mine: str = "A") -> dict:
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    wav = work / "audio_16k.wav"
    if not wav.exists():
        extract_wav(str(video), str(wav))
    sr, x = read_wav(str(wav))
    le, bg, lb = loudness(sr, x)
    t, amp = detect_hits(le, bg)
    rallies = group_rallies(t, amp)
    if not rallies:
        start = 0.0
    else:
        start, _ev = warmup_cutoff(rallies, None)
    fl = edit_flags("US")
    dur = float(video_duration(str(video)))
    cap = float(max_seconds) if max_seconds is not None else dur
    clips = rallies_only(le, bg, lb, start, fl, None, min(dur, cap), None)
    points = []
    for i, c in enumerate(clips):
        a = float(c["A"]); b = float(c["B"])
        if a >= cap:
            break
        if b - a < 0.2:
            continue
        points.append({
            "id": f"p{i+1}",
            "idx": i + 1,
            "start": round(a, 3),
            "end": round(min(b, cap), 3),
            "p_a": 0.5,
            "tier": "abstain",
            "reason": "abstain",
            "rules": [],
            "hits": int(c.get("hits") or 0),
        })
    elapsed = time.perf_counter() - t0
    used = min(dur, cap)
    hours = max(1e-9, used / 3600.0)
    return {
        "video_id": video_id,
        "video": {
            "name": video_id,
            "licence": license_,
            "url": url,
            "attribution": attribution,
        },
        "match": {
            "sport": sport,
            "preset": preset,
            "first_server": first_server,
            "mine": mine,
            "source_licence": license_,
        },
        "points": points,
        "meta": {
            "segmentation_s": round(elapsed, 2),
            "video_duration_s": round(used, 2),
            "seconds_per_video_hour": round(elapsed / hours, 2),
            "n_points": len(points),
            "method": "editloop.rallies_only (audio smart); pw_rules abstain (no tracks)",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--sport", default="padel")
    ap.add_argument("--preset", default="golden_point")
    ap.add_argument("--license", default="CC BY")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--url", default="")
    ap.add_argument("--attribution", default="")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--first-server", default="A")
    a = ap.parse_args(argv)
    vid = a.video_id or a.video.stem
    work = a.work or (a.out.parent / f"work_{vid}")
    spec = build(a.video, work, a.sport, a.preset, a.license, vid, a.url, a.attribution,
                 a.max_seconds, first_server=a.first_server)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"wrote": str(a.out), **spec["meta"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
