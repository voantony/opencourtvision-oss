#!/usr/bin/env python3
"""Cut a CC video into short per-point clips and a least-confident queue.

  python tools/labeling/export_clips.py --video match.mp4 --candidates points.json --out ~/label-session

points.json:
  {
    "video_id": "match-1",
    "video": {"name": "match-1", "licence": "CC BY", "url": "...", "attribution": "..."},
    "match": {"sport": "tennis", "preset": "advantage", "first_server": "A", "mine": "A",
              "source_licence": "CC BY"},
    "points": [{"id": "p3", "idx": 3, "start": 12.0, "end": 20.5, "p_a": 0.47,
                "tier": "abstain", "reason": "abstain", "rules": []}]
  }

Each clip is 480p H.264 with up to 1080p (no upscale), CRF 20, ~4 s before the first hit and ~2 s after. queue.json is ordered by the
decoder ask_order (least confident first) and keeps p(A) plus the rule or abstain reason.
The labeler does not show that guess until a point has an answer.

ffmpeg must be on PATH. Python stdlib plus this repo. Works with the Windows and macOS ffmpeg builds.
Synthetic footage is refused unless --allow-synthetic (tests only). Nothing here is committed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAD_S = 1.0
PRE_ROLL_S = 4.0
POST_ROLL_S = 2.0
MAX_HEIGHT = 1080
CRF = 20
MIN_SOURCE_HEIGHT = 720

ALLOWED = {"CC0", "CC BY", "CC0 1.0", "CC BY 4.0", "CC BY 3.0", "CC BY 2.0"}


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg not on PATH")
    return found


def ffprobe_bin() -> str:
    found = shutil.which("ffprobe")
    if not found:
        raise SystemExit("ffprobe not on PATH")
    return found


def duration_s(path: Path) -> float:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def order_points(points: list, sport: str, preset: str, first_server: str) -> list:
    """Decoder ask_order: least confident first. The candidate order is not trusted."""
    from opencourtvision.scoring.decoder import PointEvidence, decode
    from opencourtvision.scoring.formats import get_format

    evidence = [PointEvidence(p_a=float(p.get("p_a", 0.5)), p_point=1.0) for p in points]
    decoded = decode(evidence, get_format(sport, preset), first_server=first_server)
    ordered = []
    for i in decoded.ask_order():
        row = dict(points[i])
        row["confidence"] = float(decoded.confidence[i])
        row["reason"] = row.get("reason") or row.get("tier") or "abstain"
        row["rules"] = list(row.get("rules") or [])
        ordered.append(row)
    return ordered


def _safe(name: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "_" for c in str(name)]
    text = "".join(keep).strip("_")
    return text or "point"


def probe_wh(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


def scale_filter(src_h: int) -> str | None:
    """Keep source resolution up to 1080p; never upscale; 720p sources stay 720p."""
    if src_h <= MAX_HEIGHT:
        return None  # stream-copy scale not needed; re-encode at native size
    return f"scale=-2:{MAX_HEIGHT}"


def cut_clip(src: Path, dest: Path, start: float, end: float, src_h: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src_h is None:
        _, src_h = probe_wh(src)
    vf = scale_filter(src_h)
    cmd = [ffmpeg_bin(), "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed for {dest.name}: {proc.stderr[-500:]}")


def _licence(spec: dict) -> str:
    video = spec.get("video") or {}
    match = spec.get("match") or {}
    return (video.get("licence") or match.get("source_licence") or "").upper().replace("-", " ")


def export_session(video: Path, spec: dict, out: Path, pad: float | None = None,
                   pre_roll: float = PRE_ROLL_S, post_roll: float = POST_ROLL_S,
                   allow_synthetic: bool = False, clips_dirname: str = "clips") -> dict:
    lic = _licence(spec)
    if lic == "SYNTHETIC" and not allow_synthetic:
        raise SystemExit("refusing synthetic footage without --allow-synthetic")
    if lic not in ALLOWED and not (lic == "SYNTHETIC" and allow_synthetic):
        raise SystemExit(f"licence must be CC0 or CC BY, got {lic or 'nothing'}")
    points = list(spec.get("points") or [])
    if not points:
        raise SystemExit("candidates have no points")
    match = spec["match"]
    ordered = order_points(points, match.get("sport") or "tennis", match.get("preset") or "advantage",
                           match.get("first_server") or "A")
    dur = duration_s(video)
    video_id = str(spec.get("video_id") or spec.get("video", {}).get("name") or video.stem)
    if pad is not None:
        pre_roll = post_roll = float(pad)
    src_w, src_h = probe_wh(video)
    clips = out / clips_dirname
    chrono = sorted(ordered, key=lambda p: float(p["start"]))
    prev_end_by_idx = {}
    prev_end = 0.0
    for p in chrono:
        prev_end_by_idx[int(p["idx"])] = prev_end
        prev_end = max(prev_end, float(p["end"]))
    queue_points = []
    for p in ordered:
        start = max(0.0, float(p["start"]))
        end = min(dur, float(p["end"]))
        if end - start < 0.2:
            continue
        floor = float(prev_end_by_idx.get(int(p["idx"]), 0.0))
        clip_start = max(0.0, floor, start - float(pre_roll))
        clip_end = min(dur, end + float(post_roll))
        if clip_end - clip_start < 0.2:
            continue
        name = f"{_safe(video_id)}_{int(p['idx'])}.mp4"
        cut_clip(video, clips / name, clip_start, clip_end, src_h=src_h)
        queue_points.append({
            "id": str(p.get("id") or p["idx"]),
            "idx": int(p["idx"]),
            "video_id": video_id,
            "clip": f"{clips_dirname}/{name}",
            "start": start,
            "end": end,
            "clip_start": round(clip_start, 3),
            "clip_end": round(clip_end, 3),
            "pre_roll_s": float(pre_roll),
            "post_roll_s": float(post_roll),
            "pad_s": float(pre_roll),
            "source_wh": [src_w, src_h],
            "p_a": float(p.get("p_a", 0.5)),
            "confidence": float(p["confidence"]),
            "tier": p.get("tier") or "abstain",
            "reason": p.get("reason") or "abstain",
            "rules": list(p.get("rules") or []),
        })
    queue = {
        "schema": "ocv.label_queue.v1",
        "order": "ask_order",
        "video_id": video_id,
        "video": spec.get("video") or {"name": video_id, "licence": lic},
        "match": match,
        "points": queue_points,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    return queue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pad", type=float, default=None, help="symmetric pad; overrides pre/post if set")
    ap.add_argument("--pre-roll", type=float, default=PRE_ROLL_S)
    ap.add_argument("--post-roll", type=float, default=POST_ROLL_S)
    ap.add_argument("--clips-dirname", default="clips")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args(argv)
    spec = json.loads(a.candidates.read_text())
    queue = export_session(a.video, spec, a.out, pad=a.pad, pre_roll=a.pre_roll, post_roll=a.post_roll,
                           allow_synthetic=a.allow_synthetic, clips_dirname=a.clips_dirname)
    print(f"wrote {len(queue['points'])} clips to {a.out} (ask_order, least confident first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
