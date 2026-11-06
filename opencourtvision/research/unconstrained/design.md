# Unconstrained research track

Status: design + first scoreboard column (19 Oct 2026). Branch home: this package.
Never in the Vamos US artefact. Nothing ships until Legal clears each feature per region.

## Scope (Antony, via CEO)

Mine public competitor patents (SwingVision, Infinity Cube, Maiden, PlaySight, Wingfield,
Foxtenn, Sony) for ideas and improve on them. Covered:

- 3D ball trajectory (single-camera physics ODE; PR #18 `physics3d`)
- Two-phone 3D fusion (triangulation / stereo trajectory) — EU-only research flag pending Legal
- Shot speed from the fitted 3D track
- Backward-correcting score decoding (global / Viterbi-style)
- Motion-based clip boundaries
- Pose triggers
- Auto court calibration (no N1 tap flow)

**Line calls stay out** in every track (product rule, not a patent rule).

## Rules

1. Separate module (`opencourtvision/research/unconstrained`). CI symbol scan keeps N1–N5
   for Vamos US; this package and `physics3d` are on the denylist / EXCLUDE_US list.
2. Nothing ships until Legal clears each feature per region.
3. Scoreboard column **Unconstrained** sits next to EU / US / SwingVision (§2.8.4b).
4. No AGPL code (no Ultralytics YOLO, no BoxMOT).

## First entry (column seed)

| Metric (a recorded match held-out 51–101, or pooled S6) | Unconstrained (physics3d best) | EU (own ball + S6) | US (own ball + S6, D1) | SwingVision |
|---|---|---|---|---|
| SV-3D shot recall | **0.786** | — | — | 1.0 (ref) |
| SV-3D shot precision | 0.816 | — | — | — |
| Bounces / point | 0.53 | 0 | 0 | — |
| Final-bounce correct (SV) | 3/15-class | — | — | — |
| Point-winner S6 first call | **0.673** (tie with EU rally arm) | 0.673 | 0.644–0.594 | 0.83 |
| ms / frame (box, numpy) | ~4 | ~0.8 (2D) | ~0.8 | n/a |

What the unconstrained column currently buys over US: bounce geometry + 3D shot association
and the option to use next-serve overrides / global decode later. It does **not** yet beat
EU S6 on point winner. Point-winner calibration remains priority 1.

## N1–N5 note on physics3d

`physics3d` fits a real 3D ODE (gravity + drag + Magnus) and can emit court-metre
positions, bounce P(in) and speed. That **violates N2 / D2 as product outputs** in the
US/UK builds. Settled placement: keep the solver here under Unconstrained; the US build
ships only 2D image-space tracks (constant-acceleration / quadratic arcs) and never
packages this module.
