# Point labeler

Label who won every point of a match video, fast, on a laptop. The output is ground truth for the score engine
(`opencourtvision/scoring/decoder.py`) and training rows for the models (`opencourtvision.labels.points`).

```bash
python3 tools/labeler/serve.py "/path/to/match.MP4" --seed seed.json   # opens Safari at http://127.0.0.1:8765/
```

Stdlib only (macOS `python3` is enough). Safari plays iPhone/SwingVision 4K HEVC. Labels autosave to
`<video>.labels.json` next to the video (the previous save is kept as `.labels.bak.json`) and to the browser's localStorage.
You can also open `labeler.html` straight from disk, pick the video and the seed with the file inputs, and use **Export JSON**.

Seed: `make_seed.py --segments segments.csv --sv sv_points.json --video match.MP4 --out seed.json` pre-cuts the video into
our rally segments. SwingVision's winners show greyed as suggestions: they are not labels until you press A/B or Enter.
Segments with no SwingVision point show `×?` (usually a fault, let or cut-short rally).

## Keys
| key | action | key | action |
|---|---|---|---|
| A / B | team won the point (then the next one plays) | Enter | accept the grey suggestion |
| X | not a point (fault, let, warm-up) | S | flip the server from here on |
| Space | play / pause | R | replay the point |
| N or ↓ / P or ↑ | next / previous segment | U | next unlabeled |
| ← / → | −/+ 2 s (Shift: 10 s) | , / . | frame step |
| [ / ] | set the start / end to the playhead | I | insert a segment, or split this one at the playhead |
| M | merge with the next one | Delete / Backspace | delete the segment |
| Z / Shift-Z (or ⌘Z) | undo / redo | - / = | slower / faster (0.5–4×) |
| E | note | G | go to segment # |
| F | ★ favourite (any segment, also × like high fives) | V | favourites pass on / off |

The running score (games, call server-first, who serves, deuce/ad court) is computed from your labels by the same rules as
the decoder; `~` means some grey suggestions are still counted.

## Favourites pass
`python3 serve.py VIDEO --pass` (or press V / the ★ button) plays only the labelled points, resuming at the first one not
yet seen. At the end of each point it pauses 1.2 s: press F to star it (works while it plays too), N to skip ahead, P to go
back. Tick "pass incl. ×" to also play × segments. Every point shown is marked `fav_seen`, so non-favourites count as
real negatives. `L.highlight_rows(sv_points)` gives favourite-vs-not rows with the highlight ranker's features and
`ranker_vs_favorites(rows)` scores the ranker's top-N against them.

## Output (`ocv.point_labels.v1`)
Per segment: `start`, `end`, `kind` (`point` / `non_point` / null), `winner` (`A`/`B`), `winner_src`, `server`,
`server_src` (`rules` / `user`), `side`, `score_before` {sets, games, pts, tiebreak, call}, `notes`, `favorite`, `fav_seen`, plus the seed hints.
Plus `match` (teams, `mine`, `first_server`, preset) and `decoded` (score, winner sequence, counts).

```python
from opencourtvision.labels.points import load_point_labels
L = load_point_labels("match.labels.json")
L.verify()          # [] if every stored score agrees with the rules
L.final_score()     # "6-1 6-2"
L.to_evidence()     # PointEvidence with confirmed winners, for decoder evals
L.training_rows()   # flat rows for training
L.highlight_rows(sv_points)   # favourite vs not, with story/highlight.py features
```

Tests: `tests/test_point_labels.py`; `tests/test_labeler_browser.py` drives the real page in headless Chrome (needs
playwright + ffmpeg) and labels the whole dev_match match from the keyboard.

`benchmarks/eval_point_labels.py LABELS --sv sv_points.json [--stt stt_win_cache.json] --final 2-0` runs the score
decoder on the real signals against the labels (and the ranker against the favourites). Keep real labels out of git.
