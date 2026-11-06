# 4-player ID tracker (doubles)

Goal: four stable identities (A1, A2 = the user's team; B1, B2 = opponents) across a whole match, from one camera,
without faces or biometrics. Code: `opencourtvision/players/`; Mac measurement tool: `tools/player_id/`.

## Pipeline

1. **Detect** people with RF-DETR Nano (pretrained COCO, 384 px) on the full frame plus native-resolution tiles over
   the far court (detbench `rfn384_tiles`). Tile boxes cut by the far-ROI bottom edge (near
   players at the net) are dropped in favour of the full-frame box (`geometry.merge_far`); box-in-box duplicates are
   removed by containment NMS.
2. **Court mask + in play.** Foot points go to court metres through the court homography; people outside the play zone
   (court + run-off) are dropped (benches, the next court). Per side the best two by an in-play score are kept: in
   court, human-sized, and close to someone who was in play in the previous frame ("stays in play from the serve").
   A learned alternative (per-match MLP on the same features) is ablated.
3. **Cues** per detection: clothing colour (CIELAB histograms of the shirt band and the shorts band, inside the Apple
   Vision person instance mask, bands from Vision body pose), OSNet re-ID embedding (body + outfit), a learned
   per-match metric head on top of OSNet + colour, and serve **roles** at the contact (server / receiver by position,
   or by the arm-up pose, or by a learned pose-role MLP). Pose is used for body bands and roles only: no limb
   lengths, no gait, no faces.
4. **Tracker** (`tracker.IDTracker`), point by point:
   - *motion*: detections are chained into tracklets per side by court position; any ambiguous step (partners
     crossing) cuts the tracklets, so motion never carries an identity through a crossing;
   - tracklets that overlap in time on a side are different people, so each side's tracklets form 2-colourable
     components with exactly two labellings;
   - *court*: the changeover schedule (from the score) says which team is on which side;
   - remaining cues are summed per component in cascade order and the cascade stops at margin `tau`; appearance cues
     are computed on at most 3 frames per tracklet;
   - *rules*: within a set the serve order and receiving sides are fixed, so the server / receiver at each serve
     contact are matched against a rotation memory learned during the match.
5. **Labelling** (`deduce.deduce_ids`): the user taps **two** players, ideally one per team (e.g. himself and one
   opponent). His partner is the other in-play person on his side; the remaining person is the other opponent.
   The app should ask for the taps on a frame where all four are visible (two per side); if a side does not show two
   people the deduction defers to the next such frame. Fewer taps: one tap (or two on the same team) names that team,
   opponents stay anonymous ("Opponent 1/2"); no taps: teams by side only. Side switches and the serve rotation then
   keep the names consistent.

## Reproduce

The measurement tool is `tools/player_id/pid_run.py` (detect, vision, eval). Point it at your own local video, prelabels and court JSON via `--video`, `--prelabels` and `--court` (or the `PID_VIDEO`, `PID_PRELABELS` and `PID_COURT` environment variables). Optional: `--labels` / `PID_LABELS` for point labels, `--out` / `PID_OUT` for the run folder.

```bash
STEPS="detect vision eval" bash tools/player_id/run_pid.sh
python tools/player_id/pid_run.py eval --out ./pid_out \
    --video "$PID_VIDEO" --prelabels "$PID_PRELABELS" --court "$PID_COURT"
```

Outputs (crops, proxy truth, per-match models, check set) stay in the `--out` folder you give it and are never committed. Proxy truth for eval is described in `opencourtvision/players/truth.py` (rules truth at serve frames, motion-chain truth in rallies); it is not hand identity labels.

## Privacy gates for the closed EU beta (Ops V6/V7, 21 Oct 2026)

- **Appearance data is per match, in memory only.** OSNet embeddings and CIELAB colour histograms live inside one
  `IDTracker`. It returns slot ids and margins only, refuses pickling/copying, and `end_match()` (or leaving a
  `with IDTracker() as tr:` block) drops every template. A new match uses a new tracker, so nothing carries across
  matches and there is no "recognise this person" profile. Use `players.strip_appearance()` before saving any frames or
  detections. Enforced by `tests/test_players_privacy.py`, which includes a static check that nothing outside
  `players/` reads these features.
- **The research tool is not the product.** `tools/player_id/pid_run.py` caches appearance features under a
  private `--out` folder so the eval can be re-run. It refuses to run when `VAMOS_BETA=1` and is never packaged.
- **OSNet weights (V7):** taken from `VAMOS_MODEL_DIR/osnet/` (bundled) or the cache. In beta mode the only download
  allowed is from our EU host (`VAMOS_MODEL_HOST`); otherwise the OSNet cue is disabled. Google Drive is not used.
  Note that the MSMT17 training set is research-only, so Legal has to clear it before the weights ship.
