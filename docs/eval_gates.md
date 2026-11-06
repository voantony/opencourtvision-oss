# `ocv eval`: MVP hard gates, blind test and scoring formats

Source: Media Director, `mvp_edit_style.md` (draft v1, 18 Oct 2026), §8 and §9.

## Hard gates (§9.2), per sport
`ocv eval --gates edit_eval.json [--sport tennis|padel|pickleball]` (format: `examples/edit_eval.example.json`).

| Gate | Computed from | Target |
|---|---|---|
| Point-start accuracy | full-cut clip start vs labelled serve contact − pre-roll (4.0 s tennis, 3.0 s padel/pickleball) | ≤ 0.5 s on ≥ 98% |
| Clipped serves | clip start after the labelled `motion_start` | 0 |
| Cut-off endings | clip end < `t_dec` + buffer by clip type: reel_beat 0.4 s, point and highlight clips 1.0 s (full-cut clips are point clips) | ≤ 2%; 0 on reel beats with T1 `t_dec` |
| Missed / spurious points | full-cut clips vs labelled `[serve_contact, t_dec]` | ≤ 1% / ≤ 2% |
| Warmup leakage | seconds of any output inside labelled warmup | 0 |
| Wrong captions / caption precision | metrics | 0 / ≥ 0.95 |
| Wrong score on the bug | metrics | 0 |
| Camera smoothness | `jerk_p95` on the float path / v3 baseline (or the reel's stats JSON) | ≤ 0.5× |
| Dead time | reel max ¬live run; full-cut dead footage between points | ≤ 1.0 s; 0 s |
| Effects budget | slow-mo, verified punch-ins, captions per 8 s | ≤ 2, ≤ 3, ≤ 1 |
| Best point today | upload/stop → push, 1-h match, M2 Pro | ≤ 10 min |
| Consent | violations | 0 |

A gate without input prints `not measured` and the result is INCOMPLETE. Cut-off buffers per clip type are the Media
Director's decision (6).

## Blind test (§9.1)
`ocv eval --blind votes.csv [--exclude-raters antony,team]`; CSV `rater_id,output,match_id,sport,vote[,reason,killer_tag]`,
vote `ours|sv|tie` (tie = half a win). Pass = win rate ≥ target, comparisons ≥ minimum, ≥ 30 raters, and the lower end of
the two-sided 90% Wilson interval (z = 1.645) > 50%. The doc's check (n = 30 at 70% → ≈ 0.55) uses this z; one-sided 90%
(z = 1.28) would give 0.59. Consequences: 60% passes the Wilson rule only from n = 68 (so the ≥ 80 minimum binds), 70% from
n = 17, and 65% only from n = 31, which is why the Top 3 minimum is n ≥ 40 (Media Director decision 5).

## Scoring formats (`opencourtvision/scoring/formats.py`), checked 18 Oct 2026
**Padel**, FIP Rules of Padel (application 01.01.2026), Rule 1: three options. Advantage; **Star Point** (deuce 1 →
advantage 1 → deuce 2 → advantage 2 → deuce 3 = one deciding point); Golden Point (deciding point at the first deuce).
For a golden point or a Star Point, the **receiving pair chooses** the right or left side, can't switch positions, and in
mixed the receiver is the same sex as the server. The Star Point is used from 2026 on Premier Padel, CUPRA FIP Tour, FIP
Promises and FIP Beyond (FIP, 18 Dec 2025), and by USPA in all sanctioned play, replacing the golden point. World Padel
Tour no longer exists (Premier Padel replaced it after 2023). **Default for club footage: `golden_point`** (Media Director
decision 1). Per match: the user's setting wins; otherwise `match_format()` detects the rule from spoken calls ("punto de
oro" → golden point; "star point", "punto estrella", "deuce 2/3", "ventaja 1/2" → Star Point); otherwise the default.
Pro-tour footage: `fip_star_point`. `PUNTO DE ORO` / `STAR POINT` captions and the `ORO` / `STAR` bug cell appear only at a
verified deciding point (`stakes_caption`, `bug_point_cell`).

Coach O1 (`game_pressure`, `pressure_multiplier`): a deciding point is a break point (×1.3) if it decides the game for the
receiver, otherwise deuce (×1.15). Under the Star Point, advantage 1/2 to the receiver is a break point and deuce 1/2 is
deuce. M = min(1.75, nested max × 1.15 if deuce). The ranker using this still lives in `rallycut/work/hl_sanity.py`;
`story/ranking.py` has the older additive weights.
- https://www.padelfip.com/wp-content/uploads/2025/12/FIP_Rules-of-Padel-1.pdf
- https://www.padelfip.com/wp-content/uploads/2025/12/18-december_Star-Point_ENG.pdf
- https://padelusa.org/wp-content/uploads/2026/03/2026-COMPETITION-STRUCTURE-GUIDE-MAR-1.pdf

**Pickleball**: USA Pickleball 2026 Official Rulebook (reorganised in 2026; rule numbers changed) and UPA-A Rulebook v0.9
(21 Aug 2026; governs PPA and MLP).
- Side-out: only the server scores (USAP 4.A, UPA-A 5.6). In doubles both partners serve, except that at the start of each
  game only the starting server serves (USAP 5.B/5.B.2, UPA-A 5.5.2), so the first call is "0-0-2" (USAP 6.B.2, UPA-A 5.2.1).
  Games to 11 win by 2 (USAP 15.C/15.C.1, UPA-A TO 2.2); 15 or 21 also allowed. USAP team play may use win by 1, and UPA-A
  sanctioning recommends round robins to 11 win by 1.
- Rally (USAP 14.A, still **provisional** in 2026): a point on every rally, **including for the receiver at game point**
  (the 2025 freeze was removed); a two-number call; one server per side out. Allowed to 11, 15 or 21 (15.C.2), but not in
  double-elimination doubles, Golden Ticket events or Nationals. USAP decides in 2027 whether to keep it.
  UPA-A: rally scoring is a TD option (TO 2.3); Appendix E is still "In Progress".
- MLP 2026: doubles games are **side-out** to 11 win by 2. The DreamBreaker is rally scoring to 21 win by 2, with singles
  rotations every 4 rallies, and a team **must win while serving**.
- https://www.usapickleball.org/docs/rules/USAP-Official-Rulebook.pdf
- https://pbatf.org/wp-content/uploads/2026/01/2026_USA_Pickleball_Rulebook-Changes.pdf
- https://upaa.unitedpickleball.com/wp-content/uploads/2026/08/V0.9-8.24.26-UPA-A-Rulebook-.pdf
- https://upaa.unitedpickleball.com/wp-content/uploads/2026/05/2026-MLPlay-Rules-Guide-5.17.26.pdf
- https://majorleaguepickleball.co/abcs-of-mlp/

## Overlays and speed labels (SwingVision takeaways)
- `render/overlays.py: clear_on_cuts()`: every overlay (shot card, minimap, tags, speed, captions) ends at the first cut
  after it starts and is dropped if under 0.4 s remain. The score bug is the only persistent overlay.
- `speeds.py: check_speed() / speed_label()`: speed labels are suppressed (not corrected) when outside the per-sport,
  per-stroke range (e.g. a tennis serve below 50 mph) or inconsistent with the measured contact→bounce distance and time
  (peak must be ≥ 0.9× and ≤ 1.8× (tennis) / 1.6× (padel, pickleball) the mean flight speed).
