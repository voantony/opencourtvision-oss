"""Stage layout and a content-hash cache per stage.

Target layout (module that implements each stage today):
  ingest          ffmpeg.py                  probe, 16 kHz audio, keyframes
  events/audio    detection/audio.py         ball-hit onsets, rallies
  windows         windows.py                 analysis only near hits
  track/ball      reel/ball.py               HSV+motion ball tracker (+ optional TrackNet)
  track/players   pose/estimate.pick_player  one tracked player per camera end (4-player doubles tracking: TODO)
  pose            pose/                      keypoints, strokes, mechanics, footwork, syllables
  events/vision   pose/strokes.py            contact/stroke events from pose
  verify          events.py                  verify gate (confidence, independent sources, label rules)
  scoring         scoring/                   callouts, games, changeovers
  story           story/                     excitement, beats, difficult shots
  reel            reel/                      9:16 ball-follow reel
  render          render/                    lossless cuts
pipeline.py still uses its own per-file caches; new stages use StageCache. Migrating pipeline.py is a follow-up.

Cache key = sha256(stage name, stage version, params JSON, input fingerprints). File fingerprints hash size + the
first and last MiB (fast enough for 4K match files); pass full=True to hash everything.
"""
import hashlib
import json
import os

STAGES = ("ingest", "events/audio", "windows", "track/ball", "track/players", "pose", "events/vision", "verify",
          "scoring", "story", "reel", "render")
CHUNK = 1 << 20


def file_fingerprint(path, full=False):
    h = hashlib.sha256(); size = os.path.getsize(path); h.update(str(size).encode())
    with open(path, "rb") as f:
        if full or size <= 2 * CHUNK:
            for b in iter(lambda: f.read(CHUNK), b""): h.update(b)
        else:
            h.update(f.read(CHUNK)); f.seek(-CHUNK, os.SEEK_END); h.update(f.read(CHUNK))
    return h.hexdigest()


def fingerprint(x):
    """Inputs may be file paths, upstream cache keys, or JSON-serialisable values."""
    if isinstance(x, str) and os.path.isfile(x): return "file:" + file_fingerprint(x)
    return "val:" + hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


class StageCache:
    def __init__(self, work_dir):
        self.root = os.path.join(work_dir, "cache")

    def key(self, stage, inputs=(), params=None, version="1"):
        if stage not in STAGES: raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        blob = json.dumps(dict(stage=stage, version=version, params=params or {}, inputs=[fingerprint(i) for i in inputs]), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def path(self, stage, key, ext=".json"):
        d = os.path.join(self.root, stage.replace("/", "_")); os.makedirs(d, exist_ok=True)
        return os.path.join(d, key + ext)

    def run(self, stage, fn, inputs=(), params=None, version="1"):
        """Return fn()'s JSON result, computed once per distinct (stage, version, params, inputs). -> (result, key, hit)."""
        k = self.key(stage, inputs, params, version); p = self.path(stage, k)
        if os.path.exists(p):
            return json.load(open(p)), k, True
        res = fn(); tmp = p + ".tmp"
        json.dump(res, open(tmp, "w"), default=float); os.replace(tmp, p)
        return res, k, False
