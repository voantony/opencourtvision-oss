# Labeling public clips

Contributors label **CC0 or CC BY** padel and tennis clips only. Anything else (private matches, SwingVision exports, all-rights-reserved broadcasts, NC or ND licences) stays out of the repository and out of the benchmark.

No faces need to be named. Hitters are a court position, not a person.

## Allowed sources

| Licence | Allowed | What you must record |
|---|---|---|
| CC0 | yes | source URL, title, author if known, access date |
| CC BY | yes | the same, plus the attribution text the licence asks for |
| CC BY-SA | no | share-alike is a different licence; do not add it here |
| CC BY-NC, CC BY-ND, and anything non-commercial | no | |
| Unlicensed or all rights reserved | no | |

Put one block per clip in `docs/CC_ATTRIBUTION.md`. Pin the file with `sha256` in `benchmarks/public_data/clips_manifest.json`. `scripts/fetch_public_clips.py` will not download an entry that lacks a 64-character sha256. Do not commit the video or any frame.

## What to label

Schema: `docs/label_schema.json` (`ocv.public_labels.v1`).

For every rally that is a real point:

- `start` and `end` in seconds on the clip clock (the point, not the whole rally warm-up)
- `winner`: `A` or `B` (A is the near side, the side closest to the camera)
- `server`: `A` or `B`
- shots: each contact with `t` (seconds), `hitter` as a court position only (`near_deuce`, `near_ad`, `far_deuce`, `far_ad`), and no name
- bounces: `t` and `x`, `y` in court metres from the near-left corner (width toward the ad side, length toward the far baseline). Use `null` if you cannot place it.

`match` must include `sport`, `preset`, `first_server`, `mine`, and `source_licence` (`CC0` or `CC BY`).

Check the file before you rely on it:

```bash
python benchmarks/public_eval/validate_labels.py --labels my_clip.labels.json
```

The validator rejects duplicate idxs, the same idx at two times more than a second apart, overlaps longer than 2 seconds, and a score that does not replay. Then:

```bash
python benchmarks/public_eval/run_eval.py --labels my_clip.labels.json --p my_p_a.json
```

`my_p_a.json` is optional model output (`{"p_a": [0.5, ...]}` aligned to time-ordered scored points). Without a model, score the labels by review and send them so someone else can run a model against them.

## Submitting

Open a merge request with the label JSON, the manifest row, and the attribution block. No media. Sign off with `git commit -s` (see CONTRIBUTING.md).
